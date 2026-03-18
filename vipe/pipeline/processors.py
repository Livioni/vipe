# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_depth_to_depth, align_inv_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import (CachedVideoStream, FrameAttribute,
                               StreamProcessor, VideoFrame, VideoStream)
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode

logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0, image_dir: Path | None = None) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []
        self.image_dir = image_dir
        self.droid_intrinsics: torch.Tensor | None = None
        
        # 检测是否为DROID数据集并尝试加载intrinsics
        if self.image_dir is not None and "droid" in str(self.image_dir).lower():
            self._load_droid_intrinsics()

    def _load_droid_intrinsics(self) -> None:
        """从DROID数据集中加载intrinsics"""
        if self.image_dir is None:
            return
            
        # 从image_dir路径推断intrinsics文件路径
        # 例如: datasets/droid/Fri_Apr_21_17:11:41_2023/17368348/images/left
        # -> datasets/droid/Fri_Apr_21_17:11:41_2023/17368348/intrinsics/17368348_left.npy
        image_dir_path = Path(self.image_dir)
        
        # 提取序列ID和视角(left/right)
        view_name = image_dir_path.name  # "left" or "right"
        sequence_id = image_dir_path.parent.parent.name  # "17368348"
        
        # 构建intrinsics文件路径
        intrinsics_dir = image_dir_path.parent.parent / "intrinsics"
        intrinsics_file = intrinsics_dir / f"{sequence_id}_{view_name}.npy"
        
        if intrinsics_file.exists():
            try:
                # 加载3x3的相机内参矩阵
                K = np.load(intrinsics_file)
                # 提取 fx, fy, cx, cy
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                self.droid_intrinsics = torch.as_tensor([fx, fy, cx, cy]).float()
                logger.info(f"Loaded DROID intrinsics from {intrinsics_file}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            except Exception as e:
                logger.warning(f"Failed to load DROID intrinsics from {intrinsics_file}: {e}")
        else:
            logger.warning(f"DROID intrinsics file not found: {intrinsics_file}")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        # 如果已经加载了DROID intrinsics，直接使用
        if self.droid_intrinsics is not None:
            frame.intrinsics = self.droid_intrinsics
            frame.camera_type = self.camera_type
            return frame
        
        # 否则使用默认的估计方式
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [200, 200, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        # logger.info(f"Estimated intrinsics: {frame.intrinsics}")
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
        image_dir: Path | None = None,
    ) -> None:
        super().__init__(video_stream, gap_sec, image_dir)

        # 如果已经加载了DROID intrinsics，跳过GeoCalib估计
        if self.droid_intrinsics is not None:
            return

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(self.mask_phrases, sam_points_per_side=50, sam_run_gap=self.sam_run_gap)
        self.mask_expand = mask_expand

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame


class KnownDepthProcessor(StreamProcessor):
    """Load per-frame known depth from a directory and attach to frames."""

    def __init__(self, known_depth_dir: str | Path):
        self.known_depth_dir = Path(known_depth_dir)
        self.known_depth_paths = self._build_known_depth_index(self.known_depth_dir)
        logger.info(
            "Using known depth maps for SLAM from %s (%d frames indexed).",
            self.known_depth_dir,
            len(self.known_depth_paths),
        )

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        # Prefer raw frame index since file names usually follow the source frame id.
        depth_frame_idx = frame.raw_frame_idx
        if depth_frame_idx not in self.known_depth_paths:
            depth_frame_idx = frame_idx
        frame.metric_depth = self._load_known_depth(depth_frame_idx, frame.size()).to(frame.device)
        frame.information = "KnownDepth(SLAM)"
        return frame

    def _build_known_depth_index(self, depth_dir: Path) -> dict[int, Path]:
        if not depth_dir.exists():
            raise FileNotFoundError(f"Known depth directory does not exist: {depth_dir}")
        if not depth_dir.is_dir():
            raise NotADirectoryError(f"Known depth path is not a directory: {depth_dir}")

        index: dict[int, Path] = {}
        for suffix in ("*.npy", "*.npz"):
            for depth_path in sorted(depth_dir.glob(suffix)):
                frame_idx = self._parse_frame_index(depth_path.stem)
                if frame_idx is None:
                    continue
                index[frame_idx] = depth_path

        if len(index) == 0:
            raise ValueError(
                f"No supported depth files found in {depth_dir}. Expected *.npy or *.npz with numeric frame names."
            )
        # Some datasets are 1-based (1, 2, 3, ...). Normalize to 0-based keys for internal use.
        if 0 not in index and min(index) == 1:
            index = {frame_idx - 1: depth_path for frame_idx, depth_path in index.items()}
        return index

    def _parse_frame_index(self, stem: str) -> int | None:
        match = re.search(r"\d+", stem)
        if match is None:
            return None
        return int(match.group(0))

    def _load_known_depth(self, frame_idx: int, frame_size: tuple[int, int]) -> torch.Tensor:
        depth_path = self.known_depth_paths.get(frame_idx)
        if depth_path is None:
            raise KeyError(f"Missing depth file for frame {frame_idx} in {self.known_depth_dir}")

        if depth_path.suffix == ".npy":
            depth_np = np.load(depth_path)
        elif depth_path.suffix == ".npz":
            npz_data = np.load(depth_path)
            if len(npz_data.files) == 0:
                raise ValueError(f"Empty npz depth file: {depth_path}")
            depth_np = npz_data[npz_data.files[0]]
        else:
            raise ValueError(f"Unsupported known depth format: {depth_path.suffix}")

        depth = torch.from_numpy(np.asarray(depth_np)).float()
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                raise ValueError(f"Depth should be single-channel, but got shape {tuple(depth.shape)} at {depth_path}")
        if depth.ndim != 2:
            raise ValueError(f"Depth should be 2D after squeeze, but got shape {tuple(depth.shape)} at {depth_path}")

        if tuple(depth.shape) != frame_size:
            depth = torch.nn.functional.interpolate(depth[None, None], frame_size, mode="bilinear")[0, 0]
        depth = torch.where(depth > 0, depth, torch.nan)
        return depth


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_result: torch.Tensor = unpack_optional(
            self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale, bias = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    scale, bias = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale, bias)
                scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                self.cache_scale_bias = (scale, bias)

                video_inv_depth = video_depth_inv_depth * scale + bias
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


class MultiviewDepthProcessor(StreamProcessor):
    """
    Use multi-view depth model (e.g. DAv3, MapAnything, CAPA) to estimate depth map for each frame.
    To ensure that the depth maps are consistent with the SLAM map/pose (metric), we condition the depth model either with
    (a) sparse points, or (b) camera poses & intrinsics.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    (Multi-view input video frames are currently not supported)
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,                  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,       # This is found to cause jittering for some scenes due to abrupt context changes.
        known_depth_dir: str | None = None,
        align_to_slam: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe
        self.known_depth_dir = Path(known_depth_dir) if known_depth_dir else None
        self.align_to_slam = align_to_slam
        self.infill_target_pose = self.slam_output.get_view_trajectory(0)

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[VideoFrame] = []
        self.n_frames = 0
        self.known_depth_paths: dict[int, Path] = {}

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.known_depth_dir is not None:
            self.known_depth_paths = self._build_known_depth_index(self.known_depth_dir)
            logger.info(
                "Using known depth maps from %s (%d frames indexed).",
                self.known_depth_dir,
                len(self.known_depth_paths),
            )
            return

        if self.model == "mvd_dav3":
            try:
                from depth_anything_3.api import DepthAnything3
                from depth_anything_3.api import logger as dav3_logger
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
                )

            dav3_logger.level = 0  # Disable logging timing information
            self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
            self.dav3_api = self.dav3_api.cuda().eval()
        else:
            raise ValueError(f"Unsupported multi-view depth model: {self.model}")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("MultiviewDepthProcessor should not be called directly.")

    def _build_known_depth_index(self, depth_dir: Path) -> dict[int, Path]:
        if not depth_dir.exists():
            raise FileNotFoundError(f"Known depth directory does not exist: {depth_dir}")
        if not depth_dir.is_dir():
            raise NotADirectoryError(f"Known depth path is not a directory: {depth_dir}")

        index: dict[int, Path] = {}
        for suffix in ("*.npy", "*.npz"):
            for depth_path in sorted(depth_dir.glob(suffix)):
                frame_idx = self._parse_frame_index(depth_path.stem)
                if frame_idx is None:
                    continue
                index[frame_idx] = depth_path

        if len(index) == 0:
            raise ValueError(
                f"No supported depth files found in {depth_dir}. Expected *.npy or *.npz with numeric frame names."
            )
        return index

    def _parse_frame_index(self, stem: str) -> int | None:
        match = re.search(r"\d+", stem)
        if match is None:
            return None
        return int(match.group(0))

    def _load_known_depth(self, frame_idx: int, frame_size: tuple[int, int]) -> torch.Tensor:
        depth_path = self.known_depth_paths.get(frame_idx)
        if depth_path is None:
            raise KeyError(f"Missing depth file for frame {frame_idx} in {self.known_depth_dir}")

        if depth_path.suffix == ".npy":
            depth_np = np.load(depth_path)
        elif depth_path.suffix == ".npz":
            npz_data = np.load(depth_path)
            if len(npz_data.files) == 0:
                raise ValueError(f"Empty npz depth file: {depth_path}")
            depth_np = npz_data[npz_data.files[0]]
        else:
            raise ValueError(f"Unsupported known depth format: {depth_path.suffix}")

        depth = torch.from_numpy(np.asarray(depth_np)).float()
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                raise ValueError(f"Depth should be single-channel, but got shape {tuple(depth.shape)} at {depth_path}")
        if depth.ndim != 2:
            raise ValueError(f"Depth should be 2D after squeeze, but got shape {tuple(depth.shape)} at {depth_path}")

        if tuple(depth.shape) != frame_size:
            depth = torch.nn.functional.interpolate(depth[None, None], frame_size, mode="bilinear")[0, 0]
        depth = torch.where(depth > 0, depth, torch.nan)
        return depth

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame)
            yield frame

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        if self.known_depth_dir is not None:
            for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Loading known depth"):
                frame.metric_depth = self._load_known_depth(frame_idx, frame.size()).to(frame.device)
                if self.align_to_slam:
                    frame.metric_depth = self._align_depth_with_slam(frame_idx, frame, frame.metric_depth)
                    frame.information = "KnownDepth+SLAMAlign"
                else:
                    frame.information = "KnownDepth"
                yield frame
            return

        current_sliding_window: list[VideoFrame] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating multi-view depth"):
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                # Grab all neighboring keyframes to anchor the current sliding window.
                # Note that we remove redundant keyframes that already exist in the current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds])
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                # Perform inference
                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
                    process_res_method="lower_bound_resize",  # Keep aspect ratio
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                # Linearly interpolate the trailing depth with new depth
                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    if self.align_to_slam:
                        frame.metric_depth = self._align_depth_with_slam(
                            current_sliding_window_idx[sw_idx], frame, frame.metric_depth
                        )
                        frame.information = "MVD+SLAMAlign"
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")

    def _align_depth_with_slam(self, frame_idx: int, frame: VideoFrame, source_depth: torch.Tensor) -> torch.Tensor:
        if self.slam_output.slam_map is None:
            return source_depth
        if frame.intrinsics is None or frame.camera_type is None:
            return source_depth

        slam_depth = self.slam_output.slam_map.project_map(
            frame_idx,
            0,
            frame.size(),
            unpack_optional(frame.intrinsics),
            self.infill_target_pose[frame_idx],
            unpack_optional(frame.camera_type),
            infill=False,
        )
        valid_mask = slam_depth > 0
        if frame.mask is not None:
            valid_mask = valid_mask & frame.mask
        if frame.sky_mask is not None:
            valid_mask = valid_mask & (~frame.sky_mask)
        if valid_mask.sum().item() < 64:
            return source_depth

        try:
            return align_depth_to_depth(source_depth, slam_depth, valid_mask, quantile_masking=True, bias=True)
        except RuntimeError:
            return source_depth
