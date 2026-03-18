#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferJob:
    image_dir: Path
    output_dir: Path


LOG_FILE_NAME = "vipe_infer.log"


def parse_gpus(gpus: str) -> list[str]:
    ids = [x.strip() for x in gpus.split(",") if x.strip()]
    if not ids:
        raise ValueError("No valid GPU ids provided.")
    return ids


def discover_jobs(datasets_root: Path) -> list[InferJob]:
    jobs: list[InferJob] = []
    for left_dir in datasets_root.rglob("left"):
        # Match .../images/left
        if left_dir.is_dir() and left_dir.parent.name == "images":
            output_dir = left_dir.parent.parent
            jobs.append(InferJob(image_dir=left_dir, output_dir=output_dir))
    jobs.sort(key=lambda j: str(j.image_dir))
    return jobs


def run_one_job(job: InferJob, pipeline: str, gpu_id: str, dry_run: bool, skip_if_done: bool) -> tuple[bool, str]:
    if skip_if_done and (job.output_dir / "intrinsics").exists():
        return True, f"[GPU {gpu_id}] SKIP {job.image_dir} (intrinsics exists)"

    cmd = [
        sys.executable,
        "-m",
        "vipe.cli.main",
        "infer",
        "--image-dir",
        str(job.image_dir),
        "--pipeline",
        pipeline,
        "--output",
        str(job.output_dir),
    ]

    if dry_run:
        return True, f"[GPU {gpu_id}] DRYRUN {' '.join(cmd)}"

    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = job.output_dir / LOG_FILE_NAME
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"===== START GPU {gpu_id} =====\n")
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
        f.write(f"===== END GPU {gpu_id} (exit={ret.returncode}) =====\n")

    ok = ret.returncode == 0
    status = "OK" if ok else f"FAIL({ret.returncode})"
    return ok, f"[GPU {gpu_id}] {status} {job.image_dir}"


def worker(gpu_id: str, jobs: list[InferJob], pipeline: str, dry_run: bool, skip_if_done: bool) -> tuple[int, int]:
    succ = 0
    fail = 0
    for job in jobs:
        ok, msg = run_one_job(job, pipeline, gpu_id, dry_run, skip_if_done)
        print(msg, flush=True)
        if ok:
            succ += 1
        else:
            fail += 1
    return succ, fail


def clear_previous_logs(jobs: list[InferJob]) -> int:
    removed = 0
    for output_dir in {job.output_dir for job in jobs}:
        for log_path in output_dir.glob("vipe_infer*.log"):
            if log_path.is_file():
                log_path.unlink()
                removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run VIPE infer across multiple GPUs (one worker per GPU)."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/mnt/local/lihao/phs/EgoExpert_benchmark/ropedia/datasets"),
        help="Root folder to search for .../images/left",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="dav3_slam_only",
        help="VIPE pipeline name",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip job if output_dir/intrinsics already exists",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")

    gpu_ids = parse_gpus(args.gpus)
    jobs = discover_jobs(datasets_root)
    if not jobs:
        print(f"No jobs found under {datasets_root} (expecting .../images/left)")
        return

    print(f"Found {len(jobs)} jobs under {datasets_root}")
    print(f"Using GPUs: {gpu_ids}")

    if not args.dry_run:
        removed_logs = clear_previous_logs(jobs)
        print(f"Cleared {removed_logs} previous log files.")

    # One worker per GPU; each worker handles strided jobs.
    sharded: list[tuple[str, list[InferJob]]] = [
        (gpu_id, jobs[idx::len(gpu_ids)]) for idx, gpu_id in enumerate(gpu_ids)
    ]

    total_succ = 0
    total_fail = 0
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(worker, gpu_id, gpu_jobs, args.pipeline, args.dry_run, args.skip_if_done)
            for gpu_id, gpu_jobs in sharded
            if gpu_jobs
        ]
        for fut in as_completed(futures):
            succ, fail = fut.result()
            total_succ += succ
            total_fail += fail

    print(f"Done. success={total_succ}, fail={total_fail}, total={len(jobs)}")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
