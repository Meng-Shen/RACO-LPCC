#!/usr/bin/env python3
"""Resumable per-frame, per-rate G-PCC measurement for SemanticKITTI.

Each successful rate is appended and fsynced immediately.  On restart, valid
(scene_id, rate_id) rows are retained and only missing jobs are encoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))

from data_utils.geometry.inout import write_ply_o3d  # noqa: E402
from extension.gpcc_geo import gpcc_encode  # noqa: E402


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)
SCALES = tuple(1.0 / value for value in QSTEPS_MM)
FIELDS = (
    "scene_id", "sample_idx", "filename", "dataset_index", "rate_id",
    "qstep_mm", "position_quantization_scale", "posQuantscale", "scale",
    "num_points", "bits", "bpp", "enc_time",
)


@contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(devnull)
        os.close(saved_fd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--merge-root", default="")
    parser.add_argument("--expected-frames", type=int, default=19130)
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Split contains duplicate scene IDs")
    return tokens


def load_valid_rows(path: Path, token_to_index: dict[str, int]) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    seen = set()
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    token = row["scene_id"]
                    rate_id = int(row["rate_id"])
                    bits = int(row["bits"])
                    points = int(row["num_points"])
                    dataset_index = int(row["dataset_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (token, rate_id)
                if (
                    token not in token_to_index
                    or dataset_index != token_to_index[token]
                    or not 0 <= rate_id < len(QSTEPS_MM)
                    or bits <= 0
                    or points <= 0
                    or key in seen
                ):
                    continue
                seen.add(key)
                rows.append({field: row.get(field, "") for field in FIELDS})
    except csv.Error:
        pass
    return rows


def atomic_write_csv(path: Path, rows: list[dict]):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_write_json(path: Path, payload: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def encode_one(
    token: str,
    coords_scaled: np.ndarray,
    rate_id: int,
    tmp_dir: Path,
    cfg: Path,
):
    worker = f"{os.getpid()}_{token}_{rate_id}"
    ply = tmp_dir / f"{worker}.ply"
    bitstream = tmp_dir / f"{worker}.bin"
    write_ply_o3d(str(ply), coords_scaled, normal=True, knn=16)
    try:
        with suppress_stderr():
            log = gpcc_encode(
                str(ply), str(bitstream), posQuantscale=SCALES[rate_id],
                cfgdir=str(cfg))
        if not bitstream.is_file() or bitstream.stat().st_size <= 0:
            raise RuntimeError(f"G-PCC produced no bitstream for {token} L{rate_id}")
        elapsed = (
            float(log.get("Processing time (wall)", 0.0))
            if isinstance(log, dict) else 0.0
        )
        return bitstream.stat().st_size * 8, elapsed
    finally:
        for path in (ply, bitstream):
            if path.exists():
                path.unlink()


def run_shard(args):
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard specification")
    points_dir = Path(args.points_dir).resolve()
    split_file = Path(args.split_file).resolve()
    output = Path(args.output).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    cfg = Path(args.cfg).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tokens = read_split(split_file)
    token_to_index = {token: index for index, token in enumerate(tokens)}
    indices = list(range(args.shard_id, len(tokens), args.num_shards))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    # Sanitize a possibly interrupted final line before opening in append mode.
    rows = load_valid_rows(output, token_to_index)
    atomic_write_csv(output, rows)
    completed = {(row["scene_id"], int(row["rate_id"])) for row in rows}
    status_path = output.with_suffix(".status.json")
    started = time.time()
    completed_at_start = len(completed)

    with output.open("a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        for ordinal, dataset_index in enumerate(indices, 1):
            token = tokens[dataset_index]
            missing = [
                rate_id for rate_id in range(len(QSTEPS_MM))
                if (token, rate_id) not in completed
            ]
            if missing:
                bin_path = points_dir / f"{token}.bin"
                if not bin_path.is_file():
                    raise FileNotFoundError(bin_path)
                points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
                coords_mm = np.rint(points[:, :3].astype(np.float64) * 1000).astype(np.int32)
                coords_scaled = coords_mm - coords_mm.min(axis=0)
                num_points = len(points)
                if num_points <= 0:
                    raise RuntimeError(f"Empty frame: {bin_path}")
                for rate_id in missing:
                    bits, enc_time = encode_one(
                        token, coords_scaled, rate_id, tmp_dir, cfg)
                    qstep = QSTEPS_MM[rate_id]
                    scale = SCALES[rate_id]
                    row = {
                        "scene_id": token,
                        "sample_idx": token,
                        "filename": token,
                        "dataset_index": dataset_index,
                        "rate_id": rate_id,
                        "qstep_mm": qstep,
                        "position_quantization_scale": f"{scale:.15g}",
                        "posQuantscale": f"{scale:.15g}",
                        "scale": f"{scale:.15g}",
                        "num_points": num_points,
                        "bits": bits,
                        "bpp": f"{bits / num_points:.9f}",
                        "enc_time": f"{enc_time:.6f}",
                    }
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add((token, rate_id))

            if ordinal == 1 or ordinal % args.log_every == 0 or ordinal == len(indices):
                payload = {
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "assigned_frames": len(indices),
                    "visited_frames": ordinal,
                    "completed_rows": len(completed),
                    "new_rows_this_run": len(completed) - completed_at_start,
                    "last_scene_id": token,
                    "elapsed_seconds_this_run": time.time() - started,
                    "complete": ordinal == len(indices),
                    "output": str(output),
                }
                atomic_write_json(status_path, payload)
                print(json.dumps(payload), flush=True)


def merge(args):
    root = Path(args.merge_root).resolve()
    output = Path(args.output).resolve()
    tokens = read_split(Path(args.split_file).resolve())
    token_to_index = {token: index for index, token in enumerate(tokens)}
    paths = sorted(root.glob("shard_*/gpcc.csv"))
    if not paths:
        raise FileNotFoundError(f"No shard CSV files below {root}")
    rows = []
    seen = set()
    for path in paths:
        for row in load_valid_rows(path, token_to_index):
            key = (row["scene_id"], int(row["rate_id"]))
            if key in seen:
                raise RuntimeError(f"Duplicate merged key: {key}")
            seen.add(key)
            rows.append(row)
    expected = args.expected_frames * len(QSTEPS_MM)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["rate_id"])))
    expected_keys = [
        (tokens[index], rate_id)
        for index in range(args.expected_frames)
        for rate_id in range(len(QSTEPS_MM))
    ]
    actual_keys = [(row["scene_id"], int(row["rate_id"])) for row in rows]
    if actual_keys != expected_keys:
        raise RuntimeError("Merged rows do not exactly cover the split")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output, rows)

    averages = []
    for rate_id, qstep in enumerate(QSTEPS_MM):
        current = [row for row in rows if int(row["rate_id"]) == rate_id]
        bits = sum(int(row["bits"]) for row in current)
        points = sum(int(row["num_points"]) for row in current)
        averages.append({
            "rate_id": rate_id,
            "qstep_mm": qstep,
            "num_frames": len(current),
            "total_points": points,
            "total_bits": bits,
            "bpp": bits / points,
        })
    average_path = output.with_name("semantickitti_train_gpcc_average.csv")
    with average_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=averages[0].keys())
        writer.writeheader()
        writer.writerows(averages)
    atomic_write_json(output.with_suffix(".manifest.json"), {
        "dataset": "SemanticKITTI",
        "split": "train (sequences 00-07,09-10)",
        "frames": args.expected_frames,
        "rows": len(rows),
        "qsteps_mm_coarse_to_fine": list(QSTEPS_MM),
        "bitrate_definition": "sum(encoded bits) / sum(original frame points)",
        "decode_performed": False,
        "output": str(output),
        "average_output": str(average_path),
    })
    print(f"Merged {len(rows)} rows into {output}", flush=True)


def main():
    args = parse_args()
    if args.merge_root:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
