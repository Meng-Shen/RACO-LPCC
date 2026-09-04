#!/usr/bin/env python3
"""Train a point-cloud proxy with five task-loss heads and one six-rate bpp head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from train_cost_proxy import SparseCostProxyNet, voxelize_points


QSTEPS_MM = (128, 96, 64, 48, 32, 16)
LEVEL_ORDER = (4, 3, 2, 1, 0)  # five loss heads, fine-to-coarse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def path_key(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def load_labels(loss_csv: str, bpp_csv: str, dataset_format: str):
    loss_id = "frame_id" if dataset_format == "kitti" else "scene_id"
    if dataset_format == "kitti":
        bpp_id = "filename"
    elif dataset_format == "nuscenes":
        bpp_id = "sample_token"
    else:
        bpp_id = "scene_id"
    loss_df = pd.read_csv(loss_csv, dtype={loss_id: str}).set_index(loss_id)
    bpp_rows = list(csv.DictReader(Path(bpp_csv).open(newline="")))
    if dataset_format == "nuscenes":
        bpp_table = {
            (path_key(row["lidar_path"]), int(row["rate_id"])): row
            for row in bpp_rows
        }
    else:
        bpp_table = {(row[bpp_id], int(row["rate_id"])): row for row in bpp_rows}
    labels = {}
    for scene_id, row in loss_df.iterrows():
        scene_id = str(scene_id)
        lidar_path = str(row.get("lidar_path", ""))
        bpp_key = path_key(lidar_path) if dataset_format == "nuscenes" else scene_id
        loss_by_level = np.asarray(
            [float(row[f"L{i}_signed_delta"]) for i in range(6)], dtype=np.float32
        )
        if abs(float(loss_by_level[5])) > 1e-6:
            raise ValueError(f"L5 must be zero for {scene_id}")
        bpp = np.asarray(
            [float(bpp_table[(bpp_key, level)]["bpp"]) for level in range(6)],
            dtype=np.float32,
        )
        points = int(bpp_table[(bpp_key, 0)]["num_points"])
        if not np.isfinite(loss_by_level).all() or not np.isfinite(bpp).all():
            raise ValueError(f"Non-finite labels for {scene_id}")
        if np.any(np.diff(bpp) < -1e-7):
            raise ValueError(f"Non-monotonic bpp labels for {scene_id}: {bpp}")
        labels[scene_id] = (loss_by_level, bpp, points, lidar_path)
    return labels


class RateAwareScanNetDataset(Dataset):
    def __init__(
        self,
        points_dir,
        split_file,
        loss_csv,
        bpp_csv,
        target_scale,
        voxel_size,
        point_cloud_range,
        max_voxels,
        training,
        jitter_std,
        dataset_format,
    ):
        self.points_dir = Path(points_dir)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        self.training = bool(training)
        self.jitter_std = float(jitter_std)
        self.target_scale = float(target_scale)
        self.dataset_format = dataset_format
        labels = load_labels(loss_csv, bpp_csv, dataset_format)
        self.items = []
        for scene_id in read_split(split_file):
            if scene_id not in labels:
                raise KeyError(f"Missing labels for {scene_id}")
            loss_by_level, bpp, points, lidar_path = labels[scene_id]
            if self.dataset_format == "nuscenes":
                point_path = Path(lidar_path)
                if not point_path.is_absolute():
                    point_path = self.points_dir / point_path
            else:
                point_path = self.points_dir / f"{scene_id}.bin"
            if not point_path.is_file():
                raise FileNotFoundError(point_path)
            self.items.append((scene_id, point_path, loss_by_level, bpp, points))
        grid = np.floor(
            (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        ).astype(np.int32)
        self.spatial_shape = grid[[2, 1, 0]].tolist()
        self.num_point_features = 7
        self.mean_log_bpp = np.mean(
            np.log1p(np.stack([item[3] for item in self.items])), axis=0
        ).astype(np.float32)
        loss_targets = np.stack([
            item[2][list(LEVEL_ORDER)] * self.target_scale for item in self.items
        ]).astype(np.float32)
        # Each quantization level has a very different loss range.  Normalize
        # residuals per head so that learning is driven by frame-to-frame
        # variation at every level instead of the absolute scale of coarse
        # quantization losses.
        self.loss_head_scale = np.std(loss_targets, axis=0).clip(1e-3).astype(
            np.float32
        )
        print(
            f"Dataset {split_file}: samples={len(self.items)} "
            f"spatial_shape={self.spatial_shape} "
            f"loss_head_scale={self.loss_head_scale.tolist()}", flush=True
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene_id, path, loss_by_level, bpp, points_count = self.items[index]
        raw = np.fromfile(path, dtype=np.float32)
        point_width = {"kitti": 4, "scannet": 6, "nuscenes": 5}[
            self.dataset_format
        ]
        if raw.size % point_width:
            raise ValueError(f"Invalid point file: {path}")
        xyz = raw.reshape(-1, point_width)[:, :3].copy()
        if self.dataset_format != "kitti" and len(xyz) != points_count:
            raise ValueError(f"Point-count mismatch for {scene_id}")
        if self.training and self.jitter_std > 0:
            xyz += np.random.normal(0.0, self.jitter_std, xyz.shape).astype(np.float32)
        points = np.concatenate(
            [xyz, np.zeros((len(xyz), 1), dtype=np.float32)], axis=1
        )
        features, coords = voxelize_points(
            points,
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
            max_voxels=self.max_voxels,
            use_abs_xyz=True,
            include_intensity=False,
        )
        head_target = np.asarray(
            [[loss_by_level[level] * self.target_scale] for level in LEVEL_ORDER],
            dtype=np.float32,
        )
        return {
            "scene_id": scene_id,
            "voxel_features": torch.from_numpy(features),
            "voxel_coords": torch.from_numpy(coords),
            "loss_head_target": torch.from_numpy(head_target),
            "loss_by_level": torch.from_numpy(loss_by_level.copy()),
            "bpp_by_level": torch.from_numpy(bpp.copy()),
            "num_points": points_count,
        }


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding or duplicate samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def collate_batch(batch):
    features, coords = [], []
    for batch_index, item in enumerate(batch):
        current = item["voxel_coords"].int()
        column = torch.full((len(current), 1), batch_index, dtype=torch.int32)
        features.append(item["voxel_features"].float())
        coords.append(torch.cat([column, current], dim=1))
    return {
        "scene_id": [item["scene_id"] for item in batch],
        "voxel_features": torch.cat(features),
        "voxel_coords": torch.cat(coords).int(),
        "loss_head_target": torch.stack([item["loss_head_target"] for item in batch]),
        "loss_by_level": torch.stack([item["loss_by_level"] for item in batch]),
        "bpp_by_level": torch.stack([item["bpp_by_level"] for item in batch]),
        "num_points": torch.tensor([item["num_points"] for item in batch]),
        "batch_size": len(batch),
    }


def predictions_by_level(values):
    result = values.new_zeros((values.shape[0], 6))
    for head_index, level in enumerate(LEVEL_ORDER):
        result[:, level] = values[:, head_index, 0]
    return result


class RateAwareSparseProxy(nn.Module):
    def __init__(self, spatial_shape, feat_dim, mean_log_bpp):
        super().__init__()
        self.base = SparseCostProxyNet(
            input_channels=7,
            spatial_shape=spatial_shape,
            feat_dim=feat_dim,
            num_cost_heads=5,
            num_targets=1,
            cost_nonnegative=False,
            monotonic_cost=False,
        )
        self.rate_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(feat_dim, 6),
        )
        mean_log_bpp = torch.as_tensor(mean_log_bpp, dtype=torch.float32)
        increments = torch.diff(torch.cat([mean_log_bpp.new_zeros(1), mean_log_bpp]))
        increments = increments.clamp_min(1e-4)
        self.register_buffer("mean_log_increments", increments)
        final = self.rate_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self._global_feature = None
        self.base.global_mlp.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._global_feature = output

    def forward(self, voxel_features, voxel_coords, batch_size):
        self._global_feature = None
        output = self.base(voxel_features, voxel_coords, batch_size)
        if self._global_feature is None:
            raise RuntimeError("Global feature hook did not run")
        # Predict bounded multiplicative residuals around the empirical mean
        # log-rate curve.  This keeps all six BPP values positive/monotonic and
        # prevents expm1 from overflowing during early scratch training.
        raw_rate_residual = self.rate_head(self._global_feature)
        rate_multiplier = torch.exp(0.9 * torch.tanh(raw_rate_residual))
        positive_log_increments = self.mean_log_increments[None, :] * rate_multiplier
        rate_log_pred = torch.cumsum(positive_log_increments, dim=1)
        bpp_pred = torch.expm1(rate_log_pred).clamp_min(0.0)
        return {
            "cost_pred": output["cost_pred"],
            "rate_log_pred": rate_log_pred,
            "bpp_pred": bpp_pred,
        }


def rd_levels(loss_by_level, bpp_by_level, lambdas):
    rate_saving = bpp_by_level[:, 5:6] - bpp_by_level
    scores = (
        loss_by_level[:, None, :]
        - lambdas[None, :, None] * rate_saving[:, None, :]
    )
    return scores.argmin(dim=-1), scores


def objective(
    output, batch, lambdas, target_scale, rate_weight, rd_weight, temperature,
    loss_head_scale,
):
    cost_pred = output["cost_pred"]
    loss_head_target = batch["loss_head_target"]
    normalized_loss_residual = (
        cost_pred - loss_head_target
    ) / loss_head_scale[None, :, None]
    loss_regression = F.smooth_l1_loss(
        normalized_loss_residual, torch.zeros_like(normalized_loss_residual)
    )
    target_rate_log = torch.log1p(batch["bpp_by_level"])
    rate_regression = F.smooth_l1_loss(output["rate_log_pred"], target_rate_log)

    predicted_loss = predictions_by_level(cost_pred) / float(target_scale)
    true_levels, true_scores = rd_levels(
        batch["loss_by_level"], batch["bpp_by_level"], lambdas
    )
    _, predicted_scores = rd_levels(predicted_loss, output["bpp_pred"], lambdas)
    decision_loss = F.cross_entropy(
        (-predicted_scores / float(temperature)).reshape(-1, 6),
        true_levels.reshape(-1),
    )
    # The router is trained as a calibrated predictor.  Lambda-dependent routing
    # is deliberately kept out of the optimization objective and is evaluated
    # only after the loss and rate predictions have been produced.
    total = loss_regression + float(rate_weight) * rate_regression
    return total, loss_regression, rate_regression, decision_loss, true_levels, predicted_scores


def make_loader(
    args, split, loss_csv, bpp_csv, training,
    distributed=False, rank=0, world_size=1,
):
    dataset = RateAwareScanNetDataset(
        args.points_dir, split, loss_csv, bpp_csv, args.target_scale,
        args.voxel_size, args.point_cloud_range, args.max_voxels,
        training, args.jitter_std, args.dataset_format,
    )
    sampler = None
    if distributed:
        if training:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = DistributedEvalSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        # 1201 training scenes with batch_size=4 leave one item; dropping that
        # item avoids an invalid single-sample BatchNorm update. Validation is
        # always complete.
        drop_last=training,
        collate_fn=collate_batch,
    )
    return loader, dataset, sampler


def run_epoch(
    model, loader, device, lambdas, args, loss_head_scale,
    optimizer=None, rank=0,
):
    training = optimizer is not None
    model.train(training)
    sums = np.zeros(8 + len(lambdas), dtype=np.float64)
    progress = tqdm(
        loader,
        desc="train" if training else "val",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    for batch in progress:
        for key in ("loss_head_target", "loss_by_level", "bpp_by_level"):
            batch[key] = batch[key].to(device, non_blocking=True)
        features = batch["voxel_features"].to(device, non_blocking=True)
        coords = batch["voxel_coords"].to(device, non_blocking=True)
        n = int(batch["batch_size"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(features, coords, n)
            total, loss_reg, rate_reg, decision, true_levels, predicted_scores = objective(
                output, batch, lambdas, args.target_scale, args.rate_weight,
                args.rd_weight, args.selection_temperature, loss_head_scale,
            )
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        with torch.no_grad():
            predicted_loss = predictions_by_level(output["cost_pred"]) / args.target_scale
            predicted_levels = predicted_scores.argmin(dim=-1)
            true_chosen_score = torch.gather(
                (batch["loss_by_level"][:, None, :] + lambdas[None, :, None] * batch["bpp_by_level"][:, None, :]),
                2, predicted_levels[:, :, None]
            ).squeeze(-1)
            optimal_score = (
                batch["loss_by_level"][:, None, :]
                + lambdas[None, :, None] * batch["bpp_by_level"][:, None, :]
            ).min(dim=-1).values
            correct = (predicted_levels == true_levels).sum(dim=0).cpu().numpy()
            sums[0] += n
            sums[1] += float(total) * n
            sums[2] += float(loss_reg) * n
            sums[3] += float(rate_reg) * n
            sums[4] += float(decision) * n
            sums[5] += float(torch.abs(predicted_loss - batch["loss_by_level"]).mean()) * n
            sums[6] += float(torch.abs(output["bpp_pred"] - batch["bpp_by_level"]).mean()) * n
            sums[7] += float((true_chosen_score - optimal_score).mean()) * n
            sums[8:] += correct
        if rank == 0:
            progress.set_postfix(loss=float(total), bpp_mae=sums[6] / max(1.0, sums[0]))
    if dist.is_available() and dist.is_initialized():
        reduced = torch.as_tensor(sums, dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        sums = reduced.cpu().numpy()
    count = max(1.0, sums[0])
    accuracy = (sums[8:] / count).tolist()
    return {
        "samples": int(sums[0]),
        "total_loss": sums[1] / count,
        "loss_regression": sums[2] / count,
        "rate_regression": sums[3] / count,
        "decision_loss": sums[4] / count,
        "loss_mae": sums[5] / count,
        "bpp_mae": sums[6] / count,
        "rd_regret": sums[7] / count,
        "selection_accuracy": accuracy,
        "mean_selection_accuracy": float(np.mean(accuracy)),
    }


def append_metrics(path: Path, epoch: int, split: str, metrics: dict):
    row = {"epoch": epoch, "split": split, **metrics}
    row["selection_accuracy"] = json.dumps(row["selection_accuracy"])
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def flexible_load(model, state):
    target = model.state_dict()
    restored = {}
    for key, value in state.items():
        destination = target[key]
        if value.shape == destination.shape:
            restored[key] = value
        elif value.ndim == 5:
            # spconv 1.x/2.x may expose either [out,kx,ky,kz,in] or
            # [kx,ky,kz,in,out].  First handle checkpoints already using the
            # destination layout, then fall back to the historical transpose.
            if (
                value.shape[:4] == destination.shape[:4]
                and value.shape[4] == 8
                and destination.shape[4] == 7
            ):
                candidate = value[..., [0, 1, 2, 4, 5, 6, 7]].contiguous()
            else:
                candidate = value.permute(1, 2, 3, 4, 0).contiguous()
            # The earlier KITTI geometry checkpoint kept an all-zero
            # reflectance slot in its 8-D voxel feature layout.  The current
            # geometry-only loader removes that slot and produces 7-D features
            # (rel_xyz, density, abs_xyz).  Preserve every learned geometry
            # coefficient while dropping only the obsolete input channel.
            if (
                candidate.ndim == destination.ndim == 5
                and candidate.shape[:3] == destination.shape[:3]
                and candidate.shape[3] == 8
                and destination.shape[3] == 7
                and candidate.shape[4] == destination.shape[4]
            ):
                candidate = candidate[:, :, :, [0, 1, 2, 4, 5, 6, 7], :]
            if candidate.shape != destination.shape:
                raise RuntimeError(
                    f"Cannot restore {key}: source={tuple(value.shape)} "
                    f"converted={tuple(candidate.shape)} target={tuple(destination.shape)}"
                )
            restored[key] = candidate
        else:
            raise RuntimeError(f"Cannot restore {key}")
    with torch.no_grad():
        for key, value in restored.items():
            target[key].copy_(value.to(target[key].device, target[key].dtype))


def load_pretrained_base(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint
    for field in ("model", "model_state", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and field in state and isinstance(state[field], dict):
            state = state[field]
            break
    # A newer rate-aware checkpoint contains the same ``base`` module plus a
    # learned six-rate head.  Restore all learned tensors from that model, but
    # retain the destination dataset's empirical mean-rate increments buffer.
    full_rate_aware = any(key.startswith("base.") for key in state) and any(
        key.startswith("rate_head.") for key in state
    )
    if full_rate_aware:
        target = model.state_dict()
        normalized = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key == "mean_log_increments":
                continue
            if key in target:
                normalized[key] = value
        required = {
            key for key in target
            if key != "mean_log_increments"
        }
        missing = sorted(required - set(normalized))
        if missing:
            raise RuntimeError(
                f"Rate-aware initialization misses {len(missing)} tensors: "
                f"{missing[:8]}"
            )
        flexible_load(model, normalized)
        print(
            f"Initialized shared backbone, five loss heads, and learned BPP "
            f"head from {checkpoint_path} ({len(normalized)} tensors); kept "
            "nuScenes mean-rate increments", flush=True,
        )
        return

    target = model.base.state_dict()
    normalized = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("base."):
            key = key[len("base."):]
        if key in target:
            normalized[key] = value
    missing = sorted(set(target) - set(normalized))
    if missing:
        raise RuntimeError(
            f"Initialization checkpoint misses {len(missing)} base tensors: {missing[:8]}"
        )
    flexible_load(model.base, normalized)
    print(
        f"Initialized shared backbone and five loss heads from {checkpoint_path} "
        f"({len(normalized)} tensors)", flush=True,
    )


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, metrics, args, lambdas,
    loss_head_scale,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "lambdas": [float(value) for value in lambdas.cpu()],
            "loss_head_scale": [float(value) for value in loss_head_scale.cpu()],
            "loss_normalization": "per-head training-set standard deviation",
            "model_type": "five_loss_heads_plus_one_six_rate_bpp_head",
            "routing_rule": "argmin DeltaL-lambda*(R_high-R_q)",
        },
        path,
    )


def export_predictions(model, loader, device, lambdas, args, path):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="export-val", dynamic_ncols=True):
            for key in ("loss_head_target", "loss_by_level", "bpp_by_level"):
                batch[key] = batch[key].to(device, non_blocking=True)
            output = model(
                batch["voxel_features"].to(device),
                batch["voxel_coords"].to(device),
                int(batch["batch_size"]),
            )
            loss_pred = predictions_by_level(output["cost_pred"]) / args.target_scale
            proxy_levels, _ = rd_levels(loss_pred, output["bpp_pred"], lambdas)
            oracle_levels, _ = rd_levels(batch["loss_by_level"], batch["bpp_by_level"], lambdas)
            for index, scene_id in enumerate(batch["scene_id"]):
                row = {"scene_id": scene_id, "num_points": int(batch["num_points"][index])}
                for level in range(6):
                    row[f"L{level}_predicted_delta"] = float(loss_pred[index, level])
                    row[f"L{level}_true_delta"] = float(batch["loss_by_level"][index, level])
                    row[f"L{level}_predicted_bpp"] = float(output["bpp_pred"][index, level])
                    row[f"L{level}_true_bpp"] = float(batch["bpp_by_level"][index, level])
                for lam_index, lam in enumerate(lambdas):
                    row[f"lambda_{lam_index}"] = float(lam)
                    row[f"lambda_{lam_index}_predicted_level"] = int(proxy_levels[index, lam_index])
                    row[f"lambda_{lam_index}_oracle_level"] = int(oracle_levels[index, lam_index])
                rows.append(row)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True)
    parser.add_argument("--train-loss-csv", required=True)
    parser.add_argument("--val-loss-csv", required=True)
    parser.add_argument("--train-bpp-csv", required=True)
    parser.add_argument("--val-bpp-csv", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--test-split", default="")
    parser.add_argument("--test-loss-csv", default="")
    parser.add_argument("--test-bpp-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--dataset-format", choices=["scannet", "kitti", "nuscenes"],
        default="scannet"
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--lambdas", type=float, nargs=6, required=True)
    parser.add_argument("--target-scale", type=float, default=0.05)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.16, 0.16, 0.16])
    parser.add_argument("--point-cloud-range", type=float, nargs=6, default=[-3, -2, -1, 18, 19, 7])
    parser.add_argument("--max-voxels", type=int, default=50000)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--rd-weight", type=float, default=0.0,
                        help="Deprecated compatibility option; routing is inference-only")
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    set_seed(args.seed + rank)
    device = torch.device("cuda", local_rank)
    out = Path(args.out_dir).resolve()
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)
        run_args = {**vars(args), "world_size": world_size,
                    "global_batch_size": args.batch_size * world_size}
        (out / "args.json").write_text(json.dumps(run_args, indent=2))
    if distributed:
        dist.barrier()

    train_loader, train_dataset, train_sampler = make_loader(
        args, args.train_split, args.train_loss_csv, args.train_bpp_csv, True,
        distributed, rank, world_size,
    )
    val_loader, _, _ = make_loader(
        args, args.val_split, args.val_loss_csv, args.val_bpp_csv, False,
        distributed, rank, world_size,
    )
    model = RateAwareSparseProxy(
        train_dataset.spatial_shape, args.feat_dim, train_dataset.mean_log_bpp
    ).to(device)
    if args.init_checkpoint:
        load_pretrained_base(model, args.init_checkpoint)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    lambdas = torch.tensor(args.lambdas, dtype=torch.float32, device=device)
    loss_head_scale = torch.tensor(
        train_dataset.loss_head_scale, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metrics_path = out / "metrics.csv"
    best_score = -float("inf")
    best_epoch = 0
    best_origin = "normalized_validation_loss"
    if args.init_checkpoint:
        if is_main:
            shutil.copy2(args.init_checkpoint, out / "candidate_init.pth")
            print(
                "Saved initialization as an mAP-BPP checkpoint candidate; "
                "its historical unnormalized loss is not compared with the "
                "new normalized validation loss.", flush=True,
            )
    if distributed:
        dist.barrier()
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, lambdas, args, loss_head_scale,
            optimizer=optimizer, rank=rank,
        )
        val_metrics = run_epoch(
            model, val_loader, device, lambdas, args, loss_head_scale, rank=rank
        )
        scheduler.step()
        # Early stopping follows prediction fidelity, not any chosen lambda.
        score = -val_metrics["total_loss"]
        bare_model = model.module if isinstance(model, DistributedDataParallel) else model
        if is_main:
            append_metrics(metrics_path, epoch, "train", train_metrics)
            append_metrics(metrics_path, epoch, "val", val_metrics)
            print(
                f"Epoch {epoch:03d}: val_loss_mae={val_metrics['loss_mae']:.4f} "
                f"val_bpp_mae={val_metrics['bpp_mae']:.4f} "
                f"val_acc={val_metrics['mean_selection_accuracy']:.4f} "
                f"val_regret={val_metrics['rd_regret']:.4f}", flush=True
            )
            save_checkpoint(
                out / "latest.pth", bare_model, optimizer, scheduler,
                epoch, val_metrics, args, lambdas, loss_head_scale,
            )
            save_checkpoint(
                out / "checkpoints" / f"epoch_{epoch:03d}.pth",
                bare_model, optimizer, scheduler, epoch, val_metrics, args,
                lambdas, loss_head_scale,
            )
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_origin = "ddp_training"
            if is_main:
                save_checkpoint(
                    out / "best.pth", bare_model, optimizer, scheduler,
                    epoch, val_metrics, args, lambdas, loss_head_scale,
                )
                print(f"Saved best epoch {epoch}: score={score:.6f}", flush=True)
        else:
            stale += 1
        if distributed:
            dist.barrier()
        if stale >= args.patience:
            if is_main:
                print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not is_main:
        return

    model = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint = torch.load(out / "best.pth", map_location="cpu")
    flexible_load(model, checkpoint["model"])
    val_export_loader, _, _ = make_loader(
        args, args.val_split, args.val_loss_csv, args.val_bpp_csv, False
    )
    export_predictions(
        model, val_export_loader, device, lambdas, args,
        out / "val_rate_aware_predictions.csv"
    )
    if args.test_split:
        if not args.test_loss_csv or not args.test_bpp_csv:
            raise ValueError("--test-loss-csv and --test-bpp-csv are required with --test-split")
        test_loader, _, _ = make_loader(
            args, args.test_split, args.test_loss_csv, args.test_bpp_csv, False
        )
        export_predictions(
            model, test_loader, device, lambdas, args,
            out / "test_rate_aware_predictions.csv",
        )
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_origin": best_origin,
        "elapsed_seconds": time.time() - started,
        "best_metrics": checkpoint["metrics"],
        "lambdas": args.lambdas,
        "world_size": world_size,
        "per_gpu_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "init_checkpoint": args.init_checkpoint,
        "loss_head_scale": [float(value) for value in loss_head_scale.cpu()],
        "loss_normalization": "per-head training-set standard deviation",
        "mAP_BPP_candidates": str(out / "checkpoints"),
        "model_type": "five_loss_heads_plus_one_six_rate_bpp_head",
        "routing_rule": "argmin_q DeltaL(q)-lambda*(R_high-R_q)",
    }
    (out / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
