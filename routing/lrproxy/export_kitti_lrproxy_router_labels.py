#!/usr/bin/env python3
"""Export analytical KITTI routing decisions from a LRproxy checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from gpu_voxelizer import voxelize_batch_gpu
from lrproxy import (
    LRProxy,
    count_parameters,
    select_xyz_features,
)
from train_kitti_lrproxy_router_ddp import pack_voxel_features


NUM_LEVELS = 6


def read_ids(path: Path, limit: int = 0):
    values = [line.strip().zfill(6) for line in path.read_text().splitlines() if line.strip()]
    return values[:limit] if limit > 0 else values


class RawPointDataset(Dataset):
    def __init__(self, points_dir: Path, ids):
        self.points_dir, self.ids = points_dir, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        frame_id = self.ids[index]
        raw = np.fromfile(self.points_dir / f"{frame_id}.bin", dtype=np.float32)
        return frame_id, torch.from_numpy(raw.reshape(-1, 4)[:, :3].copy())


def collate_raw(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def strip_module(state):
    return {(key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--points-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="lrproxy")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world > 7:
        raise RuntimeError(f"node-233 is capped at GPUs 0-6; got {world}")
    device = torch.device("cuda", local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = strip_module(checkpoint["model"])
    model = LRProxy(
        256, checkpoint["loss_scales"], checkpoint["mean_log_bpp"]
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    lambdas = torch.tensor(checkpoint["lambdas"], dtype=torch.float32, device=device)
    training_args = checkpoint["args"]
    voxel_size = training_args["voxel_size"]
    point_cloud_range = training_args["point_cloud_range"]
    max_voxels = int(training_args["max_voxels"])

    all_ids = read_ids(args.split, args.max_frames)
    shard_ids = all_ids[rank::world]
    loader = DataLoader(
        RawPointDataset(args.points_dir, shard_ids), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        collate_fn=collate_raw, persistent_workers=False,
    )
    shard_path = args.output_dir / f"predictions_rank_{rank:02d}.csv"
    fields = (["frame_id"]
              + [f"pred_loss_L{i}" for i in range(NUM_LEVELS)]
              + [f"pred_bpp_L{i}" for i in range(NUM_LEVELS)]
              + [f"selected_level_rate_{i}" for i in range(len(lambdas))])
    with shard_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, (frame_ids, point_clouds) in enumerate(loader):
                point_clouds = [cloud.to(device, non_blocking=True) for cloud in point_clouds]
                features, coords = voxelize_batch_gpu(
                    point_clouds, voxel_size, point_cloud_range, max_voxels,
                    use_abs_xyz=True, include_intensity=False, random_subsample=False,
                )
                features = select_xyz_features(features)
                packed, valid_mask, lengths = pack_voxel_features(
                    features, coords, len(point_clouds), False
                )
                if packed.shape[-1] != 3:
                    raise RuntimeError(f"LRproxy expected 3 features, got {packed.shape[-1]}")
                output = model(packed, valid_mask)
                selected = torch.argmin(
                    output["loss_pred"][:, None, :]
                    + lambdas[None, :, None] * output["bpp_pred"][:, None, :], dim=-1
                )
                losses = output["loss_pred"].cpu().numpy()
                rates = output["bpp_pred"].cpu().numpy()
                selected_np = selected.cpu().numpy()
                for item, frame_id in enumerate(frame_ids):
                    row = {"frame_id": frame_id}
                    row.update({f"pred_loss_L{i}": float(losses[item, i])
                                for i in range(NUM_LEVELS)})
                    row.update({f"pred_bpp_L{i}": float(rates[item, i])
                                for i in range(NUM_LEVELS)})
                    row.update({f"selected_level_rate_{i}": int(selected_np[item, i])
                                for i in range(len(lambdas))})
                    writer.writerow(row)
                if batch_index == 0:
                    print(json.dumps({
                        "first_batch": True,
                        "rank": rank,
                        "world": world,
                        "frames": len(frame_ids),
                        "active_voxels": int(sum(lengths)),
                        "feature_dim": int(packed.shape[-1]),
                        "bpp_monotonic_violations": int(
                            (torch.diff(output["bpp_pred"], dim=1) < 0).sum()
                        ),
                    }), flush=True)

    if distributed:
        dist.barrier()
    if rank == 0:
        by_id = {}
        for shard_rank in range(world):
            with (args.output_dir / f"predictions_rank_{shard_rank:02d}.csv").open(
                newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    by_id[str(row["frame_id"]).zfill(6)] = row
        missing = [frame_id for frame_id in all_ids if frame_id not in by_id]
        if missing:
            raise RuntimeError(f"Missing LRproxy predictions: {missing[:5]}")
        manifest = {
            "mode": "lrproxy_six_absolute_pvrcnn_losses_plus_monotonic_bpp",
            "model_alias": "LRproxy",
            "input_feature_dim": 3,
            "input_feature_semantics": "normalized voxel-mean absolute XYZ (3)",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "split": str(args.split.resolve()),
            "num_frames": len(all_ids),
            "routing_rule": "argmin predicted_absolute_loss + lambda * predicted_BPP",
            "qsteps_mm_coarse_to_fine": checkpoint["qsteps_mm"],
            "lambdas_low_rate_to_high_rate": checkpoint["lambdas"],
            "bpp_monotonic_violation_rate": 0.0,
            "parameter_counts": count_parameters(model),
            "label_csvs": [],
        }
        distributions = []
        for rate_id, value in enumerate(checkpoint["lambdas"]):
            label_path = args.output_dir / f"{args.prefix}_rate_{rate_id}.csv"
            counts = [0] * NUM_LEVELS
            with label_path.open("w", newline="") as handle:
                fields = ["frame_id", "jucp_label", "lambda", "predicted_loss", "predicted_bpp"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for frame_id in all_ids:
                    source = by_id[frame_id]
                    level = int(source[f"selected_level_rate_{rate_id}"])
                    counts[level] += 1
                    writer.writerow({
                        "frame_id": frame_id,
                        "jucp_label": level,
                        "lambda": float(value),
                        "predicted_loss": source[f"pred_loss_L{level}"],
                        "predicted_bpp": source[f"pred_bpp_L{level}"],
                    })
            manifest["label_csvs"].append({
                "rate_id": rate_id, "threshold": float(value),
                "lambda": float(value), "path": str(label_path.resolve()),
            })
            distributions.append({"rate_id": rate_id, "lambda": float(value), "counts": counts})
        manifest["selection_distribution_coarse_to_fine"] = distributions
        manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (args.output_dir / "EXPORT_COMPLETE.json").write_text(json.dumps({
            "status": "complete", "model_alias": "LRproxy",
            "manifest": str(manifest_path.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "frames": len(all_ids), "world_size": world,
        }, indent=2))
        print(json.dumps(manifest, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
