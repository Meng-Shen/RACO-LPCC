#!/usr/bin/env python3
"""Export sequence-08 analytical selections from a trained LRproxy router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gpu_voxelizer import voxelize_batch_gpu
from lrproxy import (
    LRProxy,
    select_xyz_features,
)
from train_semantickitti_lrproxy_router_ddp import pack_voxel_features
from semantickitti_lrproxy_training_utils import rd_levels


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)


class RawPointDataset(Dataset):
    def __init__(self, points_dir: Path, split_file: Path):
        self.points_dir = points_dir
        self.ids = [
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        frame_id = self.ids[index]
        path = self.points_dir / f"{frame_id}.bin"
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"Invalid point cloud {path}")
        xyz = raw.reshape(-1, 4)[:, :3].copy()
        return frame_id, torch.from_numpy(xyz)


def collate_raw(batch):
    return {
        "frame_ids": [item[0] for item in batch],
        "points": [item[1] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["model"].items()
    }
    voxel_size = train_args.get("voxel_size", [0.2, 0.2, 0.2])
    pc_range = train_args.get(
        "point_cloud_range", [-100, -100, -20, 100, 100, 20]
    )
    mean_log_bpp = torch.cumsum(state["mean_log_increments"].float(), 0).numpy()
    loss_scales = state["loss_scales"].float().numpy()
    model = LRProxy(
        int(train_args.get("feat_dim", 256)),
        loss_scales,
        mean_log_bpp,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    lambdas = torch.tensor(
        checkpoint["lambdas"], dtype=torch.float32, device=device
    )
    dataset = RawPointDataset(args.points_dir, args.split_file)
    if args.max_frames > 0:
        dataset.ids = dataset.ids[:args.max_frames]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_raw,
        persistent_workers=args.workers > 0,
    )

    rows = []
    selections = [[] for _ in range(6)]
    with torch.inference_mode():
        for batch in tqdm(loader, desc="LRproxy sequence-08 routing", dynamic_ncols=True):
            point_clouds = [
                points.to(device, non_blocking=True) for points in batch["points"]
            ]
            voxel_features, voxel_coords = voxelize_batch_gpu(
                point_clouds,
                voxel_size,
                pc_range,
                int(train_args.get("max_voxels", 60000)),
                use_abs_xyz=True,
                include_intensity=False,
                random_subsample=False,
            )
            voxel_features = select_xyz_features(voxel_features)
            packed, valid_mask, _ = pack_voxel_features(
                voxel_features, voxel_coords, len(point_clouds), False
            )
            output = model(packed, valid_mask)
            chosen, _ = rd_levels(output["loss_pred"], output["bpp_pred"], lambdas)
            for batch_index, frame_id in enumerate(batch["frame_ids"]):
                row = {"frame_id": frame_id}
                for level in range(6):
                    row[f"L{level}_predicted_loss"] = float(
                        output["loss_pred"][batch_index, level]
                    )
                    row[f"L{level}_predicted_bpp"] = float(
                        output["bpp_pred"][batch_index, level]
                    )
                for rate_id, multiplier in enumerate(lambdas):
                    label = int(chosen[batch_index, rate_id])
                    row[f"lambda_{rate_id}"] = float(multiplier)
                    row[f"lambda_{rate_id}_label"] = label
                    selections[rate_id].append({
                        "frame_id": frame_id,
                        "jucp_label": label,
                    })
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = args.output_dir / "sequence08_lrproxy_predictions.csv"
    with prediction_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    label_csvs = []
    for rate_id, multiplier in enumerate(checkpoint["lambdas"]):
        path = args.output_dir / f"lrproxy_lambda_{rate_id}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_id", "jucp_label"])
            writer.writeheader()
            writer.writerows(selections[rate_id])
        label_csvs.append({
            "rate_id": rate_id,
            "threshold": float(multiplier),
            "lambda": float(multiplier),
            "path": str(path.resolve()),
        })

    manifest = {
        "mode": "lrproxy_six_direct_loss_plus_monotonic_bpp",
        "ckpt": str(args.checkpoint.resolve()),
        "split_file": str(args.split_file.resolve()),
        "velodyne_dir": str(args.points_dir.resolve()),
        "qsteps_mm_coarse_to_fine": list(QSTEPS_MM),
        "selection": "argmin_q predicted_loss(q) + lambda * predicted_BPP(q)",
        "model_type": "LRproxy; six independent direct loss heads; monotonic BPP head; no decision head",
        "input_feature_dim": 3,
        "input_feature_semantics": "normalized voxel-mean absolute XYZ (3)",
        "preprocessor": "GPU geometry voxelization",
        "lambdas": [float(value) for value in checkpoint["lambdas"]],
        "label_csvs": label_csvs,
        "prediction_csv": str(prediction_csv.resolve()),
        "num_frames": len(dataset),
        "test_used_for_training_or_lambda_selection": False,
    }
    manifest_path = args.output_dir / "lrproxy_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    counts = [
        np.bincount(
            [int(row["jucp_label"]) for row in selection], minlength=6
        ).tolist()
        for selection in selections
    ]
    completion = {
        "status": "complete",
        "manifest": str(manifest_path.resolve()),
        "num_frames": len(dataset),
        "selection_counts_coarse_to_fine": counts,
        "bpp_monotonic_by_construction": True,
    }
    (args.output_dir / "EXPORT_COMPLETE.json").write_text(
        json.dumps(completion, indent=2)
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
