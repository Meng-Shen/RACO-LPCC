#!/usr/bin/env python3
"""Resumably measure exact per-object G-PCC BPP for all ShapeNet55 objects."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


POINTS = None
TMC3 = None
TIMEOUT = None


def init_worker(points_path, tmc3_path, timeout):
    global POINTS, TMC3, TIMEOUT
    POINTS = np.load(points_path, mmap_mode="r")
    TMC3 = str(tmc3_path)
    TIMEOUT = int(timeout)


def write_ply(path: Path, integer_xyz: np.ndarray):
    with path.open("w", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(integer_xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, integer_xyz, fmt="%d %d %d")


def encode_one(task):
    index, level, qstep = task
    source_points = np.asarray(POINTS[index])
    integer = np.unique(np.rint(source_points / qstep).astype(np.int32), axis=0)
    integer = integer - integer.min(axis=0, keepdims=True)
    with tempfile.TemporaryDirectory(prefix="shapenet55_gpcc_") as directory:
        directory = Path(directory)
        source = directory / "input.ply"
        bitstream = directory / "stream.bin"
        write_ply(source, integer)
        process = subprocess.run(
            [
                TMC3, "--mode=0", f"--uncompressedDataPath={source}",
                f"--compressedStreamPath={bitstream}",
                "--positionQuantizationScale=1", "--mergeDuplicatedPoints=1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=TIMEOUT,
        )
        if process.returncode:
            raise RuntimeError(
                f"tmc3 failed sample={index} level={level}: "
                f"{(process.stdout + process.stderr)[-2400:]}"
            )
        bits = bitstream.stat().st_size * 8
    return {
        "sample_index": index,
        "level": level,
        "qstep": qstep,
        "unique_points": int(len(integer)),
        "original_points": int(len(source_points)),
        "bits": int(bits),
        "bpp": float(bits / len(source_points)),
    }


def load_finished(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {
            (int(row["sample_index"]), int(row["level"]))
            for row in csv.DictReader(handle)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--tmc3", required=True)
    parser.add_argument("--qsteps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    points_path = Path(args.points).resolve()
    tmc3_path = Path(args.tmc3).resolve()
    if not tmc3_path.is_file() or not os.access(tmc3_path, os.X_OK):
        raise FileNotFoundError(f"tmc3 executable missing: {tmc3_path}")
    points = np.load(points_path, mmap_mode="r")
    qsteps = [float(item) for item in args.qsteps.split(",") if item.strip()]
    if len(qsteps) != 6:
        raise RuntimeError(f"Expected exactly six qsteps, got {qsteps}")
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
        "sample_index", "level", "qstep", "unique_points",
        "original_points", "bits", "bpp",
    ]
    mode = "a" if output.is_file() and output.stat().st_size else "w"
    started = time.time()
    completed = 0
    with output.open(mode, newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(str(points_path), str(tmc3_path), args.timeout),
        ) as executor:
            futures = {executor.submit(encode_one, task): task for task in tasks}
            for future in as_completed(futures):
                writer.writerow(future.result())
                completed += 1
                if completed % args.log_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    elapsed = time.time() - started
                    rate = completed / max(elapsed, 1e-9)
                    remaining = (len(tasks) - completed) / max(rate, 1e-9)
                    print(
                        f"completed={completed}/{len(tasks)} rate={rate:.2f}/s eta={remaining/60:.1f}min",
                        flush=True,
                    )
        handle.flush()
        os.fsync(handle.fileno())

    rows = list(csv.DictReader(output.open(newline="")))
    keys = {(int(row["sample_index"]), int(row["level"])) for row in rows}
    expected = len(points) * len(qsteps)
    if len(keys) != expected or len(rows) != expected:
        raise RuntimeError(f"Incomplete/duplicated BPP table: rows={len(rows)} unique={len(keys)} expected={expected}")
    summary = {
        "dataset": "ShapeNet55 all official train/test objects",
        "samples": int(len(points)),
        "qsteps_coarse_to_fine": qsteps,
        "rows": expected,
        "codec": str(tmc3_path),
        "bitrate_definition": "total compressed geometry bits / total original 1024 points",
        "resumable_csv": str(output),
        "elapsed_seconds_this_run": time.time() - started,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
