#!/usr/bin/env python3
"""Cache 160 mm SUN RGB-D cells represented by their original-point mean XYZ."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def read_ids(path: Path) -> list[str]:
    return [f"{int(line):06d}" for line in path.read_text().splitlines() if line.strip()]


def quantize_xyz(xyz: np.ndarray, qstep_mm: float) -> np.ndarray:
    """Use detector-aligned cells, but represent each cell by its point centroid."""
    coords_mm = np.rint(xyz.astype(np.float64) * 1000.0).astype(np.int64)
    offset_mm = coords_mm.min(axis=0)
    grid = np.rint((coords_mm - offset_mm) / float(qstep_mm)).astype(np.int64)
    _, inverse, counts = np.unique(
        grid, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, xyz.astype(np.float64, copy=False))
    return (sums / counts[:, None]).astype(np.float32)


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--qstep-mm", type=float, default=160.0)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / f"{args.split}_points_160mm.npy"
    offsets_path = args.output_dir / f"{args.split}_offsets_160mm.npy"
    scene_ids_path = args.output_dir / f"{args.split}_scene_ids.npy"
    manifest_path = args.output_dir / f"{args.split}_160mm_manifest.json"
    if all(path.is_file() for path in (points_path, offsets_path, scene_ids_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            print(json.dumps({"status": "already_complete", **manifest}, indent=2))
            return

    scene_ids = read_ids(args.split_file)
    clouds: list[np.ndarray] = []
    counts = []
    raw_counts = []
    started = time.time()
    for ordinal, scene_id in enumerate(scene_ids, 1):
        raw = np.fromfile(args.points_dir / f"{scene_id}.bin", dtype=np.float32).reshape(-1, 6)
        quantized = quantize_xyz(raw[:, :3], args.qstep_mm)
        clouds.append(quantized)
        counts.append(len(quantized))
        raw_counts.append(len(raw))
        if ordinal == 1 or ordinal % args.log_every == 0 or ordinal == len(scene_ids):
            print(json.dumps({
                "split": args.split,
                "visited": ordinal,
                "total": len(scene_ids),
                "scene_id": scene_id,
                "unique_points": len(quantized),
                "elapsed_seconds": time.time() - started,
            }), flush=True)

    offsets = np.zeros(len(clouds) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    flat_points = np.concatenate(clouds, axis=0).astype(np.float32, copy=False)
    atomic_save(points_path, flat_points)
    atomic_save(offsets_path, offsets)
    atomic_save(scene_ids_path, np.asarray(scene_ids, dtype="U6"))
    count_array = np.asarray(counts)
    manifest = {
        "status": "complete",
        "split": args.split,
        "scenes": len(scene_ids),
        "qstep_mm": args.qstep_mm,
        "quantization_rule": (
            "detector-aligned cells: rint(xyz*1000), per-scene integer minimum offset, "
            "rint(offset coordinates/qstep), unique grid; each retained point is the mean "
            "original XYZ of all points assigned to that cell"
        ),
        "raw_points": int(np.sum(raw_counts)),
        "unique_points": int(offsets[-1]),
        "point_count": {
            "mean": float(count_array.mean()),
            "median": float(np.median(count_array)),
            "p95": float(np.percentile(count_array, 95)),
            "min": int(count_array.min()),
            "max": int(count_array.max()),
        },
        "retention": float(offsets[-1] / max(np.sum(raw_counts), 1)),
        "elapsed_seconds": time.time() - started,
        "points": str(points_path.resolve()),
        "offsets": str(offsets_path.resolve()),
        "scene_ids": str(scene_ids_path.resolve()),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
