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
import pickle

from pathlib import Path

import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    AdaptiveDepthProcessor,
    GeoCalibIntrinsicsProcessor,
    KnownDepthProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)
        self.save_artifacts_flag = bool(self.out_cfg.get("save_artifacts", False))
        self.save_pose = bool(self.out_cfg.get("save_pose", True))
        self.save_mask = bool(self.out_cfg.get("save_mask", True))
        self.save_instance_phrases = bool(self.out_cfg.get("save_instance_phrases", True))
        self.save_intrinsics = bool(self.out_cfg.get("save_intrinsics", False)) or self.save_artifacts_flag
        self.save_rgb = bool(self.out_cfg.get("save_rgb", False)) or self.save_artifacts_flag
        self.save_depth = bool(self.out_cfg.get("save_depth", False)) or self.save_artifacts_flag
        self.save_meta_info = bool(self.out_cfg.get("save_meta_info", True))
        if self.init_cfg.get("known_depth_dir", None) is not None:
            if self.slam_cfg.get("keyframe_depth", None) is not None:
                logger.info("Found init.known_depth_dir; disabling slam.keyframe_depth model and using known depth for SLAM.")
                self.slam_cfg.keyframe_depth = None
            if bool(self.slam_cfg.get("optimize_intrinsics", False)):
                logger.info("Found init.known_depth_dir; disabling slam.optimize_intrinsics for metric depth constraints.")
                self.slam_cfg.optimize_intrinsics = False

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        # 尝试从video_stream中获取image_dir路径（用于DROID数据集）
        image_dir = None
        if hasattr(video_stream, 'path'):
            image_dir = video_stream.path

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type, image_dir=image_dir))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        if (known_depth_dir := self.init_cfg.get("known_depth_dir", None)) is not None:
            init_processors.append(KnownDepthProcessor(known_depth_dir))
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(
                    MultiviewDepthProcessor(
                        slam_output,
                        model=depth_align_model,
                        known_depth_dir=self.post_cfg.get("known_depth_dir", None),
                        align_to_slam=bool(self.post_cfg.get("align_to_slam", False)),
                    )
                )
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_flags_enabled = any(
                [
                    self.save_pose,
                    self.save_intrinsics,
                    self.save_rgb,
                    self.save_depth,
                    self.save_mask,
                    self.save_instance_phrases,
                ]
            )
            if artifact_flags_enabled:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(
                    artifact_path,
                    output_stream,
                    save_pose=self.save_pose,
                    save_intrinsics=self.save_intrinsics,
                    save_rgb=self.save_rgb,
                    save_depth=self.save_depth,
                    save_mask=self.save_mask,
                    save_instance_phrases=self.save_instance_phrases,
                )
                if self.save_meta_info:
                    artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
                    with artifact_path.meta_info_path.open("wb") as f:
                        pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
