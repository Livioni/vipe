#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def discover_image_dirs(datasets_root: Path) -> list[Path]:
    image_dirs: list[Path] = []
    for left_dir in datasets_root.rglob("left"):
        if left_dir.is_dir() and left_dir.parent.name == "images":
            image_dirs.append(left_dir)
    image_dirs.sort(key=str)
    return image_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VIPE serially on one GPU with a for-loop.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/mnt/local/lihao/phs/EgoExpert_benchmark/ropedia/datasets"),
        help="Root folder to search for .../images/left",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="Single GPU id to use, e.g. 0",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="dav3_slam_only",
        help="VIPE pipeline name",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip when output_dir/intrinsics exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")

    image_dirs = discover_image_dirs(datasets_root)
    if not image_dirs:
        print(f"No scenes found under {datasets_root} (expecting .../images/left)")
        return

    print(f"Found {len(image_dirs)} scenes")
    print(f"Using GPU: {args.gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    success = 0
    failed = 0

    # Serial execution with a plain for-loop.
    for idx, image_dir in enumerate(image_dirs, start=1):
        output_dir = image_dir.parent.parent
        if args.skip_if_done and (output_dir / "intrinsics").exists():
            print(f"[{idx}/{len(image_dirs)}] SKIP {image_dir} (intrinsics exists)")
            continue

        cmd = [
            sys.executable,
            "-m",
            "vipe.cli.main",
            "infer",
            "--image-dir",
            str(image_dir),
            "--pipeline",
            args.pipeline,
            "--output",
            str(output_dir),
        ]

        print(f"\n[{idx}/{len(image_dirs)}] RUN {' '.join(cmd)}")
        if args.dry_run:
            success += 1
            continue

        ret = subprocess.run(cmd, env=env, check=False)
        if ret.returncode == 0:
            success += 1
            print(f"[{idx}/{len(image_dirs)}] OK {image_dir}")
        else:
            failed += 1
            print(f"[{idx}/{len(image_dirs)}] FAIL({ret.returncode}) {image_dir}")

    print(f"\nDone. success={success}, failed={failed}, total={len(image_dirs)}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
