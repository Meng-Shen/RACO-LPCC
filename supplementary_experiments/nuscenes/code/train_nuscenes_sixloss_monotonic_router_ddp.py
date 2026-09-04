#!/usr/bin/env python3
"""Train full or Lite-S3 nuScenes routers with six absolute losses and monotonic BPP.

Only the six task-loss values and six BPP values are regressed.  Routing is
always analytical: argmin_q predicted_loss[q] + lambda * predicted_bpp[q].
Official nuScenes mAP--BPP checkpoint selection is performed by a separate
script on the held-out official validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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

from absolute_loss_monotonic_rate_proxy import (
    AbsoluteLossMonotonicRateProxy,
    count_parameters as count_full_parameters,
    rd_levels,
)
from lite_s3_absolute_loss_monotonic_rate_proxy import (
    LiteS3AbsoluteLossMonotonicRateProxy,
    count_parameters as count_lite_parameters,
)
from gpu_voxelizer import voxelize_batch_gpu


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)
NUM_LEVELS = 6


def set_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def read_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def path_key(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def load_labels(loss_csv: str | Path, bpp_csv: str | Path) -> dict:
    loss_df = pd.read_csv(loss_csv, dtype={"scene_id": str}).set_index("scene_id")
    bpp_rows = list(csv.DictReader(Path(bpp_csv).open(newline="")))
    bpp_table = {
        (path_key(row["lidar_path"]), int(row["rate_id"])): row
        for row in bpp_rows
    }
    labels = {}
    for scene_id, row in loss_df.iterrows():
        scene_id = str(scene_id)
        lidar_path = str(row["lidar_path"])
        qsteps = tuple(int(round(float(row[f"L{i}_qstep_mm"]))) for i in range(NUM_LEVELS))
        if qsteps != QSTEPS_MM:
            raise ValueError(f"Unexpected level order for {scene_id}: {qsteps}")
        # Direct per-scale detector losses, not deltas relative to level 5.
        losses = np.asarray(
            [float(row[f"L{i}_total_loss"]) for i in range(NUM_LEVELS)],
            dtype=np.float32,
        )
        key = path_key(lidar_path)
        bpp = np.asarray(
            [float(bpp_table[(key, i)]["bpp"]) for i in range(NUM_LEVELS)],
            dtype=np.float32,
        )
        points = int(bpp_table[(key, 0)]["num_points"])
        if not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError(f"Invalid absolute losses for {scene_id}: {losses}")
        if not np.isfinite(bpp).all() or np.any(np.diff(bpp) < -1e-7):
            raise ValueError(f"Invalid BPP curve for {scene_id}: {bpp}")
        labels[scene_id] = (losses, bpp, points, lidar_path)
    return labels


class NuScenesRouterDataset(Dataset):
    def __init__(
        self,
        points_dir,
        split_file,
        loss_csv,
        bpp_csv,
        voxel_size,
        point_cloud_range,
        max_voxels,
        limit=0,
    ):
        self.points_dir = Path(points_dir)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        labels = load_labels(loss_csv, bpp_csv)
        self.items = []
        split_ids = read_split(split_file)
        if int(limit) > 0:
            split_ids = split_ids[: int(limit)]
        for scene_id in split_ids:
            if scene_id not in labels:
                raise KeyError(f"Missing labels for {scene_id}")
            losses, bpp, points, lidar_path = labels[scene_id]
            point_path = Path(lidar_path)
            if not point_path.is_absolute():
                point_path = self.points_dir / point_path
            if not point_path.is_file():
                raise FileNotFoundError(point_path)
            self.items.append((scene_id, point_path, losses, bpp, points))
        grid = np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size).astype(np.int32)
        self.spatial_shape = grid[[2, 1, 0]].tolist()
        loss_matrix = np.stack([item[2] for item in self.items])
        bpp_matrix = np.stack([item[3] for item in self.items])
        self.loss_scales = np.maximum(
            np.median(loss_matrix, axis=0).astype(np.float32), np.float32(1e-3)
        )
        self.mean_log_bpp = np.mean(np.log1p(bpp_matrix), axis=0).astype(np.float32)
        print(
            f"Dataset {split_file}: samples={len(self.items)} spatial_shape={self.spatial_shape} "
            f"loss_scales={self.loss_scales.tolist()} mean_log_bpp={self.mean_log_bpp.tolist()}",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene_id, path, losses, bpp, points_count = self.items[index]
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 5:
            raise ValueError(f"Invalid nuScenes point file: {path}")
        points = raw.reshape(-1, 5)
        if len(points) != points_count:
            raise ValueError(f"Point-count mismatch for {scene_id}: {len(points)} != {points_count}")
        return {
            "scene_id": scene_id,
            "points": torch.from_numpy(points[:, :3].copy()),
            "loss_by_level": torch.from_numpy(losses.copy()),
            "bpp_by_level": torch.from_numpy(bpp.copy()),
            "num_points": points_count,
        }


class DistributedEvalSampler(Sampler):
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
    return {
        "scene_id": [item["scene_id"] for item in batch],
        "points": [item["points"] for item in batch],
        "loss_by_level": torch.stack([item["loss_by_level"] for item in batch]),
        "bpp_by_level": torch.stack([item["bpp_by_level"] for item in batch]),
        "num_points": torch.tensor([item["num_points"] for item in batch]),
        "batch_size": len(batch),
    }


def make_loader(args, split, loss_csv, bpp_csv, training, distributed=False, rank=0, world_size=1):
    dataset = NuScenesRouterDataset(
        args.points_dir,
        split,
        loss_csv,
        bpp_csv,
        args.voxel_size,
        args.point_cloud_range,
        args.max_voxels,
        args.max_train_frames if training else args.max_val_frames,
    )
    sampler = None
    if distributed:
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            if training else DistributedEvalSampler(dataset, rank, world_size)
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=collate_batch,
        persistent_workers=args.workers > 0,
    )
    return loader, dataset, sampler


def build_model(variant, spatial_shape, feat_dim, loss_scales, mean_log_bpp):
    cls = AbsoluteLossMonotonicRateProxy if variant == "full" else LiteS3AbsoluteLossMonotonicRateProxy
    return cls(
        spatial_shape=spatial_shape,
        feat_dim=feat_dim,
        loss_scales=loss_scales,
        mean_log_bpp=mean_log_bpp,
        input_channels=7,
    )


def count_parameters(model, variant):
    return count_full_parameters(model) if variant == "full" else count_lite_parameters(model)


@torch.no_grad()
def load_full_checkpoint_into_lite(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
    target = model.state_dict()
    loaded = {}
    sliced = None
    for key, destination in target.items():
        if key in {"loss_scales", "mean_log_increments", "unit_softplus_bias"}:
            continue
        value = source.get(key)
        if value is None:
            continue
        if value.shape == destination.shape:
            loaded[key] = value
        elif (
            key == "base.global_mlp.0.weight"
            and value.ndim == 2
            and value.shape[0] == destination.shape[0]
            and value.shape[1] >= destination.shape[1]
        ):
            loaded[key] = value[:, : destination.shape[1]].clone()
            sliced = {"source_shape": list(value.shape), "target_shape": list(destination.shape)}
    current = model.state_dict()
    current.update(loaded)
    model.load_state_dict(current)
    parameter_keys = set(dict(model.named_parameters()))
    loaded_parameters = sum(target[key].numel() for key in loaded if key in parameter_keys)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "source": str(checkpoint_path),
        "mode": "full_sixloss_to_lite_s3",
        "loaded_tensor_count": len(loaded),
        "loaded_parameter_count": int(loaded_parameters),
        "total_parameter_count": int(total_parameters),
        "loaded_parameter_fraction": float(loaded_parameters / total_parameters),
        "sliced_global_mlp": sliced,
    }


def prepare_points(batch, device, training, jitter_std):
    result = []
    for points in batch["points"]:
        current = points.to(device, non_blocking=True)
        if training and jitter_std > 0:
            current = current + torch.randn_like(current) * float(jitter_std)
        result.append(current)
    return result


def forward_batch(model, batch, device, args, training):
    points = prepare_points(batch, device, training, args.jitter_std)
    voxel_features, voxel_coords = voxelize_batch_gpu(
        points,
        args.voxel_size,
        args.point_cloud_range,
        args.max_voxels,
        use_abs_xyz=True,
        include_intensity=False,
        random_subsample=training,
    )
    return model(voxel_features, voxel_coords, len(points))


def run_epoch(model, loader, device, lambdas, loss_scales, args, optimizer=None, rank=0):
    training = optimizer is not None
    model.train(training)
    size = int(lambdas.numel())
    totals = np.zeros(8 + 3 * size + size * NUM_LEVELS, dtype=np.float64)
    scales = loss_scales[None, :]
    progress = tqdm(loader, desc="train" if training else "val", dynamic_ncols=True, disable=rank != 0)
    for batch_index, batch in enumerate(progress):
        loss_target = batch["loss_by_level"].to(device, non_blocking=True)
        bpp_target = batch["bpp_by_level"].to(device, non_blocking=True)
        count = int(batch["batch_size"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward_batch(model, batch, device, args, training)
            loss_reg = F.smooth_l1_loss(output["loss_pred"] / scales, loss_target / scales)
            rate_reg = F.smooth_l1_loss(output["rate_log_pred"], torch.log1p(bpp_target))
            total = args.loss_weight * loss_reg + args.rate_weight * rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        with torch.no_grad():
            predicted_levels, predicted_scores = rd_levels(output["loss_pred"], output["bpp_pred"], lambdas)
            oracle_levels, true_scores = rd_levels(loss_target, bpp_target, lambdas)
            selected_true_scores = torch.gather(true_scores, 2, predicted_levels[:, :, None]).squeeze(-1)
            optimal_scores = true_scores.min(dim=-1).values
            selected_bpp = torch.gather(bpp_target, 1, predicted_levels)
            totals[0] += count
            totals[1] += float(total) * count
            totals[2] += float(loss_reg) * count
            totals[3] += float(rate_reg) * count
            totals[4] += float(torch.abs(output["loss_pred"] - loss_target).mean()) * count
            totals[5] += float(torch.abs(output["bpp_pred"] - bpp_target).mean()) * count
            totals[6] += float((selected_true_scores - optimal_scores).mean()) * count
            totals[7] += float((torch.diff(output["bpp_pred"], dim=1) < 0).sum())
            offset = 8
            totals[offset:offset + size] += (predicted_levels == oracle_levels).sum(0).cpu().numpy()
            offset += size
            totals[offset:offset + size] += selected_bpp.sum(0).cpu().numpy()
            offset += size
            totals[offset:offset + size] += torch.gather(loss_target, 1, predicted_levels).sum(0).cpu().numpy()
            offset += size
            for rate_id in range(size):
                counts = torch.bincount(predicted_levels[:, rate_id], minlength=NUM_LEVELS)
                begin = offset + rate_id * NUM_LEVELS
                totals[begin:begin + NUM_LEVELS] += counts.cpu().numpy()
        if batch_index == 0 and rank == 0:
            print(
                f"[first_batch] mode={'train' if training else 'val'} batch={count} "
                f"points={sum(len(value) for value in batch['points'])} loss={float(total):.6f} "
                f"bpp_monotonic={bool(torch.all(torch.diff(output['bpp_pred'], dim=1) >= 0))}",
                flush=True,
            )
        if rank == 0:
            progress.set_postfix(loss=float(total), bpp_mae=totals[5] / max(totals[0], 1.0))
    packed = torch.tensor(totals, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(packed)
    values = packed.cpu().numpy()
    count = max(values[0], 1.0)
    offset = 8
    accuracy = values[offset:offset + size] / count
    offset += size
    curve_bpp = values[offset:offset + size] / count
    offset += size
    curve_loss = values[offset:offset + size] / count
    offset += size
    selection_counts = values[offset:].reshape(size, NUM_LEVELS).astype(np.int64)
    return {
        "samples": int(values[0]),
        "total_loss": values[1] / count,
        "loss_regression_normalized": values[2] / count,
        "rate_regression_log1p": values[3] / count,
        "loss_mae": values[4] / count,
        "bpp_mae": values[5] / count,
        "rd_regret": values[6] / count,
        "bpp_monotonic_violation_rate": values[7] / (count * 5.0),
        "selection_accuracy": accuracy.tolist(),
        "mean_selection_accuracy": float(accuracy.mean()),
        "curve_mean_bpp": curve_bpp.tolist(),
        "curve_mean_task_loss": curve_loss.tolist(),
        "selection_counts_coarse_to_fine": selection_counts.tolist(),
    }


def append_metrics(path, epoch, split, metrics):
    row = {"epoch": epoch, "split": split, **metrics}
    row = {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
    exists = Path(path).is_file()
    with Path(path).open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args, lambdas, init_report):
    torch.save({
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "lambdas": lambdas.detach().cpu().tolist(),
        "qsteps_mm": list(QSTEPS_MM),
        "model_variant": args.model_variant,
        "model_type": "six_direct_absolute_loss_heads_plus_monotonic_six_bpp",
        "routing_rule": "argmin_q predicted_loss(q) + lambda * predicted_BPP(q)",
        "optimization_targets": "absolute task-loss regression + BPP regression only",
        "initialization_report": init_report,
        "parameter_counts": count_parameters(model, args.model_variant),
    }, path)


@torch.no_grad()
def export_predictions(model, loader, device, lambdas, args, path):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="export", dynamic_ncols=True):
        loss_target = batch["loss_by_level"].to(device, non_blocking=True)
        bpp_target = batch["bpp_by_level"].to(device, non_blocking=True)
        output = forward_batch(model, batch, device, args, False)
        proxy_levels, _ = rd_levels(output["loss_pred"], output["bpp_pred"], lambdas)
        oracle_levels, _ = rd_levels(loss_target, bpp_target, lambdas)
        for index, scene_id in enumerate(batch["scene_id"]):
            row = {"scene_id": scene_id, "num_points": int(batch["num_points"][index])}
            for level in range(NUM_LEVELS):
                row[f"L{level}_predicted_delta"] = float(output["loss_pred"][index, level])
                row[f"L{level}_true_delta"] = float(loss_target[index, level])
                row[f"L{level}_predicted_bpp"] = float(output["bpp_pred"][index, level])
                row[f"L{level}_true_bpp"] = float(bpp_target[index, level])
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
    parser.add_argument("--model-variant", choices=["full", "lite_s3"], required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--init-kind", choices=["legacy", "full_sixloss"], default="legacy")
    parser.add_argument("--lambdas", type=float, nargs=6, required=True)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.16, 0.16, 0.16])
    parser.add_argument("--point-cloud-range", type=float, nargs=6, default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    parser.add_argument("--max-voxels", type=int, default=50000)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-weight", type=float, default=2.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank)) if distributed else 0
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world > 7:
        raise RuntimeError(f"At most seven GPUs are allowed, got {world}")
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)
    output = Path(args.out_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
    if distributed:
        dist.barrier()

    train_loader, train_set, train_sampler = make_loader(
        args, args.train_split, args.train_loss_csv, args.train_bpp_csv, True, distributed, rank, world
    )
    val_loader, _, _ = make_loader(
        args, args.val_split, args.val_loss_csv, args.val_bpp_csv, False, distributed, rank, world
    )
    model = build_model(
        args.model_variant,
        train_set.spatial_shape,
        args.feat_dim,
        train_set.loss_scales,
        train_set.mean_log_bpp,
    ).to(device)
    if args.model_variant == "lite_s3" and args.init_kind == "full_sixloss":
        init_report = load_full_checkpoint_into_lite(model, args.init_checkpoint)
    else:
        init_report = model.load_legacy_checkpoint(args.init_checkpoint)
    if rank == 0:
        run_args = {
            **vars(args),
            "world_size": world,
            "global_batch_size": args.batch_size * world,
            "loss_scales": train_set.loss_scales.tolist(),
            "mean_log_bpp": train_set.mean_log_bpp.tolist(),
            "spatial_shape": train_set.spatial_shape,
        }
        (output / "args.json").write_text(json.dumps(run_args, indent=2))
        (output / "initialization_report.json").write_text(json.dumps(init_report, indent=2))
        print(json.dumps({
            "model_variant": args.model_variant,
            "world_size": world,
            "train_frames": len(train_set),
            "val_frames": len(val_loader.dataset),
            "parameters": count_parameters(model, args.model_variant),
            "initialization": init_report,
        }, indent=2), flush=True)

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if ".cost_heads." in name or name.startswith("rate_head.") else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    lambdas = torch.tensor(args.lambdas, dtype=torch.float32, device=device)
    loss_scales = torch.tensor(train_set.loss_scales, dtype=torch.float32, device=device)
    metrics_path = output / "metrics.csv"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, lambdas, loss_scales, args, optimizer, rank)
        val_metrics = run_epoch(model, val_loader, device, lambdas, loss_scales, args, None, rank)
        scheduler.step()
        bare = model.module if isinstance(model, DistributedDataParallel) else model
        if rank == 0:
            append_metrics(metrics_path, epoch, "train", train_metrics)
            append_metrics(metrics_path, epoch, "val", val_metrics)
            save_checkpoint(output / "latest.pth", bare, optimizer, scheduler, epoch, val_metrics, args, lambdas, init_report)
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:03d}.pth", bare, optimizer, scheduler, epoch, val_metrics, args, lambdas, init_report)
            print(
                f"epoch={epoch:03d} GPUs={world} val_loss_MAE={val_metrics['loss_mae']:.6f} "
                f"val_bpp_MAE={val_metrics['bpp_mae']:.6f} RD_regret={val_metrics['rd_regret']:.6f} "
                f"monotonic_violation={val_metrics['bpp_monotonic_violation_rate']:.1f}",
                flush=True,
            )
    if distributed:
        dist.barrier()
    if rank == 0:
        summary = {
            "status": "training_complete_awaiting_official_map_bpp_selection",
            "epochs": args.epochs,
            "elapsed_seconds": time.time() - started,
            "world_size": world,
            "model_variant": args.model_variant,
            "parameter_counts": count_parameters(model.module if isinstance(model, DistributedDataParallel) else model, args.model_variant),
            "optimization_targets": "six absolute task losses and monotonic BPP only",
            "test_used_for_training": False,
        }
        (output / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
