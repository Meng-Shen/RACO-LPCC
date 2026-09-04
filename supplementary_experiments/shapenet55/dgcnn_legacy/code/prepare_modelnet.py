#!/usr/bin/env python3
"""Prepare the official ModelNet40 HDF5 split for DGCNN and routing."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def load_partition(root: Path, partition: str, num_points: int):
    files = sorted(glob.glob(str(root / f"*{partition}*.h5")))
    if not files:
        raise FileNotFoundError(f"No {partition} HDF5 shards below {root}")
    points, labels = [], []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            points.append(handle["data"][:, :num_points, :3].astype(np.float32))
            labels.append(handle["label"][:].reshape(-1).astype(np.int64))
    return np.concatenate(points), np.concatenate(labels), files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    root = Path(args.h5_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_points, train_labels, train_files = load_partition(root, "train", args.num_points)
    test_points, test_labels, test_files = load_partition(root, "test", args.num_points)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=args.seed
    )
    route_train, route_val = next(splitter.split(train_points, train_labels))
    arrays = {
        "train_points.npy": train_points,
        "train_labels.npy": train_labels,
        "test_points.npy": test_points,
        "test_labels.npy": test_labels,
        "route_train_indices.npy": route_train.astype(np.int64),
        "route_val_indices.npy": route_val.astype(np.int64),
    }
    for name, array in arrays.items():
        np.save(output / name, array)

    summary = {
        "dataset": "ModelNet40",
        "geometry_features": "XYZ only",
        "num_points": args.num_points,
        "official_train_samples": int(len(train_points)),
        "official_test_samples": int(len(test_points)),
        "router_train_samples": int(len(route_train)),
        "router_validation_samples": int(len(route_val)),
        "router_split_seed": args.seed,
        "test_used_for_training_or_selection": False,
        "train_h5_files": train_files,
        "test_h5_files": test_files,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
