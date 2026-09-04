#!/usr/bin/env python3
"""Measure per-object geometry-only G-PCC rates with resumable checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


WORKER_POINTS = None
WORKER_TMC3 = None
WORKER_TIMEOUT = None


def init_worker(points_path, tmc3_path, timeout):
    global WORKER_POINTS, WORKER_TMC3, WORKER_TIMEOUT
    WORKER_POINTS = np.load(points_path, mmap_mode="r")
    WORKER_TMC3 = str(tmc3_path)
    WORKER_TIMEOUT = int(timeout)


def write_ply(path: Path, integer_xyz: np.ndarray):
    with path.open("w", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(integer_xyz)}\n")
        # TMC3's PLY reader expects coordinate properties declared as float,
        # even though the lossless geometry values themselves are integers.
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, integer_xyz, fmt="%d %d %d")


def encode_one(task):
    index, level, qstep = task
    points = np.asarray(WORKER_POINTS[index])
    integer = np.unique(np.rint(points / qstep).astype(np.int32), axis=0)
    # Translation does not affect geometry distortion.  Moving the minimum to
    # zero also avoids relying on codec-specific handling of negative PLY ints.
    integer = integer - integer.min(axis=0, keepdims=True)
    with tempfile.TemporaryDirectory(prefix="modelnet_gpcc_") as directory:
        directory = Path(directory)
        source = directory / "input.ply"
        bitstream = directory / "stream.bin"
        write_ply(source, integer)
        command = [
            WORKER_TMC3,
            "--mode=0",
            f"--uncompressedDataPath={source}",
            f"--compressedStreamPath={bitstream}",
            "--positionQuantizationScale=1",
            "--mergeDuplicatedPoints=1",
        ]
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=WORKER_TIMEOUT,
        )
        if process.returncode:
            raise RuntimeError(
                f"tmc3 failed index={index} level={level}: "
                f"{(process.stdout + process.stderr)[-2400:]}"
            )
        bits = bitstream.stat().st_size * 8
    return {
        "sample_index": index,
        "level": level,
        "qstep": qstep,
        "unique_points": int(len(integer)),
        "original_points": int(len(points)),
        "bits": int(bits),
        "bpp": float(bits / len(points)),
    }


def parse_steps(value):
    return [float(item) for item in value.split(",") if item.strip()]


def load_finished(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {(int(row["sample_index"]), int(row["level"])) for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--partition", choices=["train", "test"], required=True)
    parser.add_argument("--tmc3", required=True)
    parser.add_argument("--qsteps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    points_path = Path(args.points).resolve()
    tmc3_path = Path(args.tmc3).resolve()
    if not tmc3_path.is_file() or not os.access(tmc3_path, os.X_OK):
        raise FileNotFoundError(f"Executable tmc3 not found: {tmc3_path}")
    points = np.load(points_path, mmap_mode="r")
    qsteps = parse_steps(args.qsteps)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    finished = load_finished(output)
    tasks = [
        (index, level, qstep)
        for index in range(len(points))
        for level, qstep in enumerate(qsteps)
        if (index, level) not in finished
    ]
    fieldnames = [
        "partition", "sample_index", "level", "qstep", "unique_points",
        "original_points", "bits", "bpp",
    ]
    mode = "a" if output.is_file() and output.stat().st_size else "w"
    started = time.time()
    completed_now = 0
    with output.open(mode, newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        if tasks:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(str(points_path), str(tmc3_path), args.timeout),
            ) as executor:
                future_map = {executor.submit(encode_one, task): task for task in tasks}
                for future in as_completed(future_map):
                    row = future.result()
                    writer.writerow({"partition": args.partition, **row})
                    completed_now += 1
                    if completed_now % args.log_every == 0:
                        handle.flush()
                        os.fsync(handle.fileno())
                        elapsed = time.time() - started
                        rate = completed_now / max(elapsed, 1e-6)
                        remaining = (len(tasks) - completed_now) / max(rate, 1e-6)
                        print(
                            f"[{completed_now}/{len(tasks)}] rate={rate:.2f}/s "
                            f"eta={remaining/60:.1f}min", flush=True
                        )
        handle.flush()
        os.fsync(handle.fileno())

    rows = list(csv.DictReader(output.open(newline="")))
    keys = {(int(row["sample_index"]), int(row["level"])) for row in rows}
    expected = len(points) * len(qsteps)
    if len(keys) != expected:
        raise RuntimeError(f"Incomplete or duplicated BPP table: {len(keys)} != {expected}")
    summary = {
        "partition": args.partition,
        "samples": int(len(points)),
        "qsteps_coarse_to_fine": qsteps,
        "rows": len(keys),
        "codec": str(tmc3_path),
        "bitrate_definition": "compressed bits / 1024 original input points",
        "elapsed_seconds_this_run": time.time() - started,
        "resumable_csv": str(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
