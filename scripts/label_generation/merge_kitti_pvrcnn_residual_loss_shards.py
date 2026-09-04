#!/usr/bin/env python3
"""Merge and validate sharded six-scale PV-RCNN loss labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def frame_ids(path: Path) -> list[str]:
    return [line.strip().zfill(6) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expected = frame_ids(args.split)
    rows_by_id: dict[str, dict[str, str]] = {}
    manifests = []
    fieldnames: list[str] = []
    for shard_id in range(args.num_shards):
        csv_path = args.shard_dir / f"shard_{shard_id}.csv"
        json_path = csv_path.with_suffix(".json")
        if not csv_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"incomplete shard {shard_id}: {csv_path}")
        manifests.append(json.loads(json_path.read_text()))
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                fid = str(row["frame_id"]).zfill(6)
                if fid in rows_by_id:
                    raise ValueError(f"duplicate frame {fid}")
                for level in range(6):
                    value = float(row[f"L{level}_total_loss"])
                    if not (0.0 <= value < float("inf")):
                        raise ValueError(f"invalid L{level} loss for {fid}: {value}")
                rows_by_id[fid] = row

    missing = [fid for fid in expected if fid not in rows_by_id]
    extra = sorted(set(rows_by_id).difference(expected))
    if missing or extra:
        raise RuntimeError(f"label coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    if not fieldnames:
        raise RuntimeError("all shard CSVs are empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_id[fid] for fid in expected)

    losses = [[float(rows_by_id[fid][f"L{i}_total_loss"]) for fid in expected] for i in range(6)]
    summary = {
        "status": "complete",
        "split": str(args.split.resolve()),
        "num_frames": len(expected),
        "num_shards": args.num_shards,
        "output": str(args.output.resolve()),
        "mean_absolute_loss_coarse_to_fine": [sum(values) / len(values) for values in losses],
        "min_absolute_loss_coarse_to_fine": [min(values) for values in losses],
        "max_absolute_loss_coarse_to_fine": [max(values) for values in losses],
        "total_shard_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in manifests),
        "max_shard_elapsed_seconds": max(float(item["elapsed_seconds"]) for item in manifests),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
