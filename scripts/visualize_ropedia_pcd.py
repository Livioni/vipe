#!/usr/bin/env python3
"""Visualize Ropedia episode point clouds frame-by-frame using viser.

Usage:
    python scripts/visualize_ropedia_pcd.py datasets/ropedia/datasets/0a5009c4-292b-40c6-b9ec-7c75cf54a112/ep9
    python scripts/visualize_ropedia_pcd.py datasets/ropedia/datasets/0a5009c4-292b-40c6-b9ec-7c75cf54a112/ep9 --port 8080 --downsample 4 --fps 1.0
"""

import argparse
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import viser
import viser.transforms as vtf


def load_episode(ep_dir: Path):
    """Load poses and intrinsics for the episode."""
    pose_data = np.load(str(ep_dir / "pose" / "left.npz"))
    poses = pose_data["data"]  # (N, 4, 4) float32 — T_world_camera
    pose_inds = pose_data["inds"]  # (N,) int64 — frame index per pose

    with h5py.File(str(ep_dir / "annotation.hdf5"), "r") as f:
        K = f["calibration/cam01/K"][:]  # [fx, fy, cx, cy] at image resolution

    image_dir = ep_dir / "images" / "left"
    n_images = len(list(image_dir.glob("frame_*_rgb.png")))
    n_frames = min(len(pose_inds), n_images)

    return poses, pose_inds, K, n_frames


def unproject_frame(
    ep_dir: Path,
    frame_idx: int,
    K: np.ndarray,
    downsample: int,
    conf_threshold: float,
    max_depth: float,
):
    """Unproject a single frame's depth to 3D points in camera space.

    Returns (points_cam, colors) where points_cam is (M, 3) and colors is (M, 3) uint8.
    """
    fname = f"frame_{frame_idx:05d}_rgb.png"

    img = cv2.cvtColor(
        cv2.imread(str(ep_dir / "images" / "left" / fname)),
        cv2.COLOR_BGR2RGB,
    )
    depth_raw = cv2.imread(
        str(ep_dir / "depths" / fname), cv2.IMREAD_UNCHANGED
    )
    conf_raw = cv2.imread(
        str(ep_dir / "conf_mask" / fname), cv2.IMREAD_UNCHANGED
    )

    depth = depth_raw.astype(np.float32) / 1000.0  # uint16 mm -> meters
    conf = conf_raw.astype(np.float32) / 65535.0  # uint16 -> [0, 1]

    H, W = depth.shape
    fx, fy, cx, cy = K

    v_idx = np.arange(0, H, downsample)
    u_idx = np.arange(0, W, downsample)
    uu, vv = np.meshgrid(u_idx, v_idx)

    d = depth[vv, uu]
    c = conf[vv, uu]
    rgb = img[vv, uu]

    valid = (d > 0.01) & (d < max_depth) & (c > conf_threshold)

    X = (uu[valid].astype(np.float32) - cx) / fx * d[valid]
    Y = (vv[valid].astype(np.float32) - cy) / fy * d[valid]
    Z = d[valid]

    return np.stack([X, Y, Z], axis=-1), rgb[valid]


def main():
    parser = argparse.ArgumentParser(
        description="Ropedia episode point-cloud viewer (viser)"
    )
    parser.add_argument("ep_dir", type=str, help="Path to an episode directory, e.g. datasets/ropedia/datasets/<uuid>/ep9")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--downsample", type=int, default=4, help="Spatial downsample factor for each frame's depth map")
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=1.0, help="Playback frames per second")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold [0,1]")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Max depth in meters")
    args = parser.parse_args()

    ep_dir = Path(args.ep_dir)
    assert ep_dir.exists(), f"Directory not found: {ep_dir}"

    poses, pose_inds, K, n_frames = load_episode(ep_dir)
    print(f"Episode: {ep_dir}")
    print(f"  {n_frames} frames | K = {K}")

    server = viser.ViserServer(port=args.port)

    # ── GUI controls ──
    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_button("Play / Pause")
        gui_prev = server.gui.add_button("<  Prev")
        gui_next = server.gui.add_button("Next  >")
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
        )
        gui_fps = server.gui.add_slider(
            "FPS", min=0.1, max=30.0, step=0.1, initial_value=args.fps
        )

    with server.gui.add_folder("Point Cloud"):
        gui_ds = server.gui.add_slider(
            "Downsample", min=1, max=32, step=1, initial_value=args.downsample
        )
        gui_ps = server.gui.add_slider(
            "Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size
        )
        gui_conf = server.gui.add_slider(
            "Conf Threshold", min=0.0, max=1.0, step=0.01, initial_value=args.conf_threshold
        )
        gui_maxd = server.gui.add_slider(
            "Max Depth (m)", min=0.5, max=5.0, step=0.5, initial_value=args.max_depth
        )
        gui_shape = server.gui.add_dropdown(
            "Point Shape",
            options=["square", "circle", "diamond", "rounded", "sparkle"],
            initial_value="circle",
        )

    with server.gui.add_folder("Display"):
        gui_show_cam = server.gui.add_checkbox("Show Camera Frustum", initial_value=True)
        gui_show_traj = server.gui.add_checkbox("Show Trajectory", initial_value=True)
        gui_cam_scale = server.gui.add_slider(
            "Frustum Scale", min=0.01, max=0.5, step=0.01, initial_value=0.1
        )

    # ── State ──
    playing = [False]
    current_frame = [0]
    need_update = [True]

    pcd_handle = [None]
    frustum_handle = [None]
    traj_handle = [None]

    # ── Callbacks ──
    @gui_play.on_click
    def _(_):
        playing[0] = not playing[0]

    @gui_prev.on_click
    def _(_):
        playing[0] = False
        current_frame[0] = max(0, current_frame[0] - 1)
        gui_frame.value = current_frame[0]
        need_update[0] = True

    @gui_next.on_click
    def _(_):
        playing[0] = False
        current_frame[0] = min(n_frames - 1, current_frame[0] + 1)
        gui_frame.value = current_frame[0]
        need_update[0] = True

    @gui_frame.on_update
    def _(_):
        new_val = int(gui_frame.value)
        if new_val != current_frame[0]:
            current_frame[0] = new_val
            need_update[0] = True

    for ctrl in [gui_ds, gui_ps, gui_conf, gui_maxd, gui_shape,
                 gui_show_cam, gui_show_traj, gui_cam_scale]:
        @ctrl.on_update
        def _(_):
            need_update[0] = True

    # ── Pre-compute trajectory for display ──
    def draw_trajectory():
        if traj_handle[0] is not None:
            traj_handle[0].remove()
            traj_handle[0] = None

        if not gui_show_traj.value:
            return

        positions = poses[:n_frames, :3, 3]
        # Subsample trajectory for performance (every 10 frames)
        step = max(1, n_frames // 500)
        traj_pts = positions[::step].astype(np.float32)
        traj_colors = np.zeros((len(traj_pts), 3), dtype=np.uint8)
        traj_colors[:, 1] = 200  # green trajectory
        traj_handle[0] = server.scene.add_point_cloud(
            "/trajectory",
            points=traj_pts,
            colors=traj_colors,
            point_size=float(gui_ps.value) * 0.5,
            point_shape="circle",
        )

    # ── Main update ──
    def update_display():
        idx = current_frame[0]
        frame_idx = int(pose_inds[idx])

        t0 = time.time()
        points_cam, colors = unproject_frame(
            ep_dir,
            frame_idx,
            K,
            downsample=int(gui_ds.value),
            conf_threshold=float(gui_conf.value),
            max_depth=float(gui_maxd.value),
        )
        load_ms = (time.time() - t0) * 1000

        pose = poses[idx]
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (points_cam @ R.T + t).astype(np.float32)

        wxyz = vtf.SO3.from_matrix(R).wxyz

        # Point cloud
        if pcd_handle[0] is not None:
            pcd_handle[0].remove()
        pcd_handle[0] = server.scene.add_point_cloud(
            "/pcd",
            points=points_world,
            colors=colors.astype(np.uint8),
            point_size=float(gui_ps.value),
            point_shape=gui_shape.value,
        )

        # Camera frustum
        if frustum_handle[0] is not None:
            frustum_handle[0].remove()
            frustum_handle[0] = None

        if gui_show_cam.value:
            fx = K[0]
            img_h = 512  # image resolution
            fov_y = float(2 * np.arctan(img_h / 2.0 / fx))
            frustum_handle[0] = server.scene.add_camera_frustum(
                "/camera",
                fov=fov_y,
                aspect=1.0,
                scale=float(gui_cam_scale.value),
                wxyz=wxyz,
                position=t.astype(np.float64),
                color=(255, 50, 50),
            )

        draw_trajectory()
        need_update[0] = False

        print(
            f"\rFrame {idx:5d}/{n_frames - 1}  |  "
            f"{len(points_world):6d} pts  |  "
            f"load {load_ms:.0f}ms",
            end="",
            flush=True,
        )

    # ── Initial render ──
    update_display()
    print(f"\nServer running at http://localhost:{args.port}")

    # ── Event loop ──
    try:
        while True:
            if playing[0]:
                current_frame[0] = (current_frame[0] + 1) % n_frames
                gui_frame.value = current_frame[0]
                update_display()
                time.sleep(1.0 / max(0.1, float(gui_fps.value)))
            elif need_update[0]:
                update_display()
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
