#!/usr/bin/env python3
"""Run a non-destructive one-batch smoke test of the nuScenes rate proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_scannet_rate_aware_proxy import (
    RateAwareSparseProxy,
    load_pretrained_base,
    make_loader,
    objective,
)


def make_loss_csv(path: Path, bpp_csv: Path, data_root: Path, count: int = 8):
    bpp = pd.read_csv(bpp_csv, dtype={"sample_token": str})
    first = (
        bpp[bpp["rate_id"] == 0]
        .sort_values("dataset_index")
        .head(count)
        .copy()
    )
    if len(first) != count:
        raise RuntimeError(f"Need {count} frames, found {len(first)}")
    rows = []
    for rank, row in enumerate(first.itertuples(index=False)):
        lidar = data_root / "samples" / "LIDAR_TOP" / Path(row.lidar_path).name
        if not lidar.is_file():
            raise FileNotFoundError(lidar)
        if lidar.stat().st_size // (5 * 4) != int(row.num_points):
            raise RuntimeError(f"Point count mismatch: {lidar}")
        # The exact values are unimportant for a plumbing test.  They preserve
        # the production schema, L5=0 baseline, and frame-to-frame variation.
        deltas = np.asarray([1.0, 0.65, 0.38, 0.20, 0.08, 0.0]) * (1 + rank / 20)
        item = {
            "scene_id": row.sample_token,
            "sample_idx": row.sample_token,
            "dataset_index": int(row.dataset_index),
            "lidar_path": str(lidar),
        }
        for level, value in enumerate(deltas):
            item[f"L{level}_signed_delta"] = float(value)
        rows.append(item)
    pd.DataFrame(rows).to_csv(path, index=False)
    return [row["scene_id"] for row in rows]


def write_split(path: Path, tokens):
    path.write_text("".join(f"{token}\n" for token in tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    work = base / "preflight" / "rate_proxy"
    work.mkdir(parents=True, exist_ok=True)
    bpp_csv = base / "labels" / "nuscenes_train_gpcc_per_frame_per_rate.csv"
    loss_csv = work / "loss.csv"
    tokens = make_loss_csv(loss_csv, bpp_csv, base / "data" / "nuscenes")
    train_split, val_split = work / "train.txt", work / "val.txt"
    write_split(train_split, tokens[:4])
    write_split(val_split, tokens[4:])

    class Args:
        points_dir = str(base / "data" / "nuscenes")
        train_loss_csv = str(loss_csv)
        val_loss_csv = str(loss_csv)
        train_bpp_csv = str(bpp_csv)
        val_bpp_csv = str(bpp_csv)
        target_scale = 1.0
        voxel_size = [0.16, 0.16, 0.16]
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        max_voxels = 50000
        jitter_std = 0.0
        dataset_format = "nuscenes"
        batch_size = 2
        workers = 0
        rate_weight = 1.0
        rd_weight = 0.0
        selection_temperature = 1.0

    train_loader, train_set = make_loader(
        Args, str(train_split), str(loss_csv), str(bpp_csv), True
    )
    val_loader, _ = make_loader(
        Args, str(val_split), str(loss_csv), str(bpp_csv), False
    )
    device = torch.device("cuda:0")
    model = RateAwareSparseProxy(
        train_set.spatial_shape, 256, train_set.mean_log_bpp
    ).to(device)
    load_pretrained_base(
        model, base / "checkpoints" / "kitti_rate_aware_5loss_plus_bpp_best.pth"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lambdas = torch.tensor([0, 0.02, 0.05, 0.1, 0.2, 0.4], device=device)
    summaries = []
    for name, loader, training in (
        ("train", train_loader, True), ("val", val_loader, False)
    ):
        batch = next(iter(loader))
        for key in ("loss_head_target", "loss_by_level", "bpp_by_level"):
            batch[key] = batch[key].to(device)
        output = model(
            batch["voxel_features"].to(device),
            batch["voxel_coords"].to(device),
            int(batch["batch_size"]),
        )
        values = objective(
            output, batch, lambdas, Args.target_scale, Args.rate_weight,
            Args.rd_weight, Args.selection_temperature,
        )
        total = values[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite {name} objective")
        bpp_pred = output["bpp_pred"].detach()
        if torch.any(torch.diff(bpp_pred, dim=1) < 0):
            raise RuntimeError("Predicted BPP is not monotonic")
        summaries.append({
            "split": name,
            "batch": int(batch["batch_size"]),
            "objective": float(total.detach()),
            "loss_shape": list(output["cost_pred"].shape),
            "bpp_shape": list(bpp_pred.shape),
        })
    result = {
        "status": "PASS",
        "checkpoint": "KITTI five-loss-heads plus BPP-head",
        "tests": summaries,
    }
    (work / "PASS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
