#!/usr/bin/env python3
"""Train three-coordinate LRproxy on all SemanticKITTI labels.

The analytical router remains argmin_q predicted_loss[q] + lambda *
predicted_bpp[q].  No decision head or decision loss is used.  Checkpoints and
early stopping use the minimum training regression loss; sequence 08 is
test-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lrproxy import (
    LRProxy,
    count_parameters,
    select_xyz_features,
)
from gpu_voxelizer import voxelize_batch_gpu
from semantickitti_lrproxy_training_utils import (
    DistributedEvalSampler,
    load_semantickitti_labels,
    normalized_curve_auc,
    rd_levels,
    read_ids,
    set_seed,
)


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)
NUM_LEVELS = 6


class SemanticKITTIRawRouterDataset(Dataset):
    """Load raw XYZ only; all augmentation and voxelization run on CUDA."""

    def __init__(
        self,
        points_dir: Path,
        split: Path | list[Path],
        loss_csv: Path,
        bpp_csv: Path,
        label_cache: Path | None,
        limit: int = 0,
    ):
        self.points_dir = Path(points_dir)
        split_paths = [split] if isinstance(split, Path) else list(split)
        ids = []
        for split_path in split_paths:
            ids.extend(read_ids(split_path))
        # Keep the split-file order while guarding against accidental overlap.
        self.ids = list(dict.fromkeys(ids))
        if limit > 0:
            self.ids = self.ids[:limit]
        self.labels = load_semantickitti_labels(loss_csv, bpp_csv, label_cache)
        for frame_id in self.ids:
            if frame_id not in self.labels:
                raise KeyError(f"Missing labels for {frame_id}")
            point_path = self.points_dir / f"{frame_id}.bin"
            if not point_path.is_file():
                raise FileNotFoundError(point_path)
        bpp_stack = np.stack([self.labels[value][1] for value in self.ids])
        self.mean_log_bpp = np.log1p(bpp_stack).mean(axis=0).astype(np.float32)
        print(
            f"Raw CUDA-voxelized dataset {[str(value) for value in split_paths]}: "
            f"{len(self.ids)} frames",
            flush=True,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        frame_id = self.ids[index]
        loss, bpp, quality, points_count = self.labels[frame_id]
        raw = np.fromfile(self.points_dir / f"{frame_id}.bin", dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"Invalid point cloud {frame_id}")
        xyz = raw.reshape(-1, 4)[:, :3].copy()
        return {
            "scene_id": frame_id,
            "points": torch.from_numpy(xyz),
            "loss_by_level": torch.from_numpy(loss.copy()),
            "bpp_by_level": torch.from_numpy(bpp.copy()),
            "quality_by_level": torch.from_numpy(quality.copy()),
            "num_points": points_count,
        }


def raw_collate(batch):
    return {
        "scene_id": [item["scene_id"] for item in batch],
        "points": [item["points"] for item in batch],
        "loss_by_level": torch.stack([item["loss_by_level"] for item in batch]),
        "bpp_by_level": torch.stack([item["bpp_by_level"] for item in batch]),
        "quality_by_level": torch.stack([item["quality_by_level"] for item in batch]),
        "num_points": torch.tensor([item["num_points"] for item in batch]),
        "batch_size": len(batch),
    }


def make_raw_loader(dataset, batch_size, workers, training, sampler):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=raw_collate,
        persistent_workers=workers > 0,
    )


def prepare_points(batch, device, training, jitter_std, rotation_aug):
    result = []
    for points in batch["points"]:
        current = points.to(device, non_blocking=True)
        if training and rotation_aug:
            angle = (torch.rand((), device=device) * 2.0 - 1.0) * math.pi
            cosine, sine = torch.cos(angle), torch.sin(angle)
            rotation = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
            current = current.clone()
            current[:, :2] = current[:, :2] @ rotation.T
        if training and jitter_std > 0:
            current = current + torch.randn_like(current) * float(jitter_std)
        result.append(current)
    return result


def pack_voxel_features(features, coords, batch_size: int, training: bool):
    """Pack flattened three-coordinate voxel features for LRproxy."""
    clouds = [features[coords[:, 0].long() == index] for index in range(batch_size)]
    lengths = [int(cloud.shape[0]) for cloud in clouds]
    if min(lengths) <= 0:
        raise RuntimeError(f"Empty LRproxy cloud in batch: {lengths}")
    padded = pad_sequence(clouds, batch_first=True, padding_value=0.0)
    sequence = torch.arange(padded.shape[1], device=features.device)
    valid_mask = sequence[None, :] < torch.as_tensor(
        lengths, device=features.device
    )[:, None]
    if training:
        for batch_index, length in enumerate(lengths):
            if length < padded.shape[1]:
                choices = torch.randint(
                    length, (padded.shape[1] - length,), device=features.device
                )
                padded[batch_index, length:] = padded[batch_index, choices]
    return padded, valid_mask, lengths


def train_only_loss_scales(dataset: SemanticKITTIRawRouterDataset) -> np.ndarray:
    values = np.stack([dataset.labels[frame_id][0] for frame_id in dataset.ids])
    # A tiny number of quantized frames improve mIoU.  A task loss is
    # non-negative, so only the regression target is clamped; signed labels are
    # retained for oracle/regret and validation curve calculations.
    nonnegative = np.maximum(values, 0.0)
    scales = np.median(nonnegative, axis=0).astype(np.float32)
    return np.maximum(scales, np.float32(1e-3))


def run_epoch(model, loader, optimizer, device, lambdas, loss_scales, args):
    training = optimizer is not None
    model.train(training)
    size = int(lambdas.numel())
    # count + six scalar sums + monotonic comparisons + three lambda vectors
    # + lambda-by-level selection counts.
    totals = np.zeros(8 + 3 * size + size * NUM_LEVELS, np.float64)
    scales = loss_scales[None, :]
    for batch_index, batch in enumerate(loader):
        for key in ("loss_by_level", "bpp_by_level", "quality_by_level"):
            batch[key] = batch[key].to(device, non_blocking=True)
        points = prepare_points(
            batch, device, training, args.jitter_std, not args.no_rotation_aug
        )
        features, coords = voxelize_batch_gpu(
            points,
            args.voxel_size,
            args.point_cloud_range,
            args.max_voxels,
            use_abs_xyz=True,
            include_intensity=False,
            random_subsample=training,
        )
        features = select_xyz_features(features)
        count = len(points)
        packed_features, valid_mask, lengths = pack_voxel_features(
            features, coords, count, training
        )
        if packed_features.shape[-1] != 3:
            raise RuntimeError(
                f"LRproxy expected 3 features, got {packed_features.shape[-1]}"
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(packed_features, valid_mask)
            regression_target = batch["loss_by_level"].clamp_min(0.0)
            loss_reg = F.smooth_l1_loss(
                output["loss_pred"] / scales,
                regression_target / scales,
            )
            rate_reg = F.smooth_l1_loss(
                output["rate_log_pred"], torch.log1p(batch["bpp_by_level"])
            )
            total = args.loss_weight * loss_reg + args.rate_weight * rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        if batch_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[first_batch] mode={'train' if training else 'val'} "
                f"batch={count} active_voxels={int(features.shape[0])} "
                f"raw_points={sum(len(value) for value in batch['points'])} "
                f"padded_shape={list(packed_features.shape)} "
                f"voxel_min={min(lengths)} voxel_max={max(lengths)} "
                f"loss={float(total):.6f}",
                flush=True,
            )
        with torch.no_grad():
            predicted_levels, _ = rd_levels(
                output["loss_pred"], output["bpp_pred"], lambdas
            )
            oracle_levels, true_scores = rd_levels(
                batch["loss_by_level"], batch["bpp_by_level"], lambdas
            )
            selected_scores = torch.gather(
                true_scores, 2, predicted_levels[:, :, None]
            ).squeeze(-1)
            optimal_scores = true_scores.min(dim=-1).values
            selected_bpp = torch.gather(
                batch["bpp_by_level"], 1, predicted_levels
            )
            selected_quality = torch.gather(
                batch["quality_by_level"], 1, predicted_levels
            )
            totals[0] += count
            totals[1] += float(total) * count
            totals[2] += float(loss_reg) * count
            totals[3] += float(rate_reg) * count
            totals[4] += float(
                torch.abs(output["loss_pred"] - batch["loss_by_level"]).mean()
            ) * count
            totals[5] += float(
                torch.abs(output["bpp_pred"] - batch["bpp_by_level"]).mean()
            ) * count
            totals[6] += float((selected_scores - optimal_scores).mean()) * count
            totals[7] += float(
                (torch.diff(output["bpp_pred"], dim=1) < 0).sum()
            )
            offset = 8
            totals[offset:offset + size] += (
                predicted_levels == oracle_levels
            ).sum(0).cpu().numpy()
            offset += size
            totals[offset:offset + size] += selected_bpp.sum(0).cpu().numpy()
            offset += size
            totals[offset:offset + size] += selected_quality.sum(0).cpu().numpy()
            offset += size
            for rate_id in range(size):
                counts = torch.bincount(
                    predicted_levels[:, rate_id], minlength=NUM_LEVELS
                )
                begin = offset + rate_id * NUM_LEVELS
                totals[begin:begin + NUM_LEVELS] += counts.cpu().numpy()

    packed = torch.tensor(totals, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(packed)
    values = packed.cpu().numpy()
    count = max(values[0], 1.0)
    offset = 8
    accuracy = values[offset:offset + size] / count
    offset += size
    mean_bpp = values[offset:offset + size] / count
    offset += size
    mean_quality = values[offset:offset + size] / count
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
        "selection_counts_coarse_to_fine": selection_counts.tolist(),
        "curve_bpp": mean_bpp.tolist(),
        "curve_framewise_miou": mean_quality.tolist(),
        "framewise_miou_bpp_auc": normalized_curve_auc(mean_bpp, mean_quality),
    }


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, metrics, args, lambdas, world,
    initialization_report,
):
    torch.save({
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "lambdas": lambdas.detach().cpu().tolist(),
        "qsteps_mm": list(QSTEPS_MM),
        "model_alias": "LRproxy",
        "model_type": "lrproxy_six_direct_loss_heads_plus_monotonic_six_bpp",
        "input_feature_dim": 3,
        "input_feature_semantics": "normalized voxel-mean absolute XYZ (3)",
        "routing_rule": "argmin_q predicted_loss(q) + lambda * predicted_BPP(q)",
        "optimization_targets": "direct task-loss regression + BPP regression only",
        "selection_metric": "minimum full-training regression total loss",
        "world_size": int(world),
        "initialization_report": initialization_report,
        "parameter_counts": count_parameters(model),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True, type=Path)
    parser.add_argument("--loss-csv", required=True, type=Path)
    parser.add_argument("--bpp-csv", required=True, type=Path)
    parser.add_argument("--label-cache", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path, nargs="+")
    parser.add_argument("--lambda-json", required=True, type=Path)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-weight", type=float, default=2.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.16, 0.16, 0.16])
    parser.add_argument(
        "--point-cloud-range", type=float, nargs=6,
        default=[-100, -100, -20, 100, 100, 20],
    )
    parser.add_argument("--max-voxels", type=int, default=60000)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--no-rotation-aug", action="store_true")
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world > 7:
        raise RuntimeError(f"This server is capped at seven GPUs, got {world}")
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "args.json").write_text(
            json.dumps(vars(args), default=str, indent=2)
        )
    if distributed:
        dist.barrier()

    common = dict(
        points_dir=args.points_dir,
        loss_csv=args.loss_csv,
        bpp_csv=args.bpp_csv,
        label_cache=args.label_cache,
    )
    train_set = SemanticKITTIRawRouterDataset(
        split=args.train_split, limit=args.max_train_frames, **common
    )
    train_sampler = (
        DistributedSampler(train_set, shuffle=True, seed=args.seed)
        if distributed else None
    )
    train_loader = make_raw_loader(
        train_set, args.batch_size, args.workers, True, train_sampler
    )
    lambda_data = json.loads(args.lambda_json.read_text())
    lambdas = torch.tensor(
        lambda_data["lambdas_high_rate_to_low_rate"],
        dtype=torch.float32,
        device=device,
    )
    loss_scale_np = train_only_loss_scales(train_set)
    loss_scales = torch.tensor(loss_scale_np, dtype=torch.float32, device=device)

    model = LRProxy(
        args.feat_dim,
        loss_scale_np,
        train_set.mean_log_bpp,
    ).to(device)
    initialization_report = model.load_full_checkpoint(args.init_checkpoint)
    if initialization_report["new_backbone_randomly_initialized"]:
        raise RuntimeError(f"LRProxy initialization was incomplete: {initialization_report}")
    if rank == 0:
        (args.output_dir / "initialization_report.json").write_text(
            json.dumps(initialization_report, indent=2)
        )
        print(json.dumps({
            "world_size": world,
            "train_frames": len(train_set),
            "validation_frames": 0,
            "training_uses_all_merged_split_frames": True,
            "loss_scales": loss_scale_np.tolist(),
            "parameters": count_parameters(model),
            "initialization": initialization_report,
        }, indent=2), flush=True)

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if ".cost_heads." in name or name.startswith("rate_head.") else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    start_epoch, best_epoch, best_score, stale = 1, 0, math.inf, 0
    resumed_from = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint["model"].items()
        }
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(
            checkpoint.get(
                "best_epoch", checkpoint["metrics"].get("best_epoch", checkpoint["epoch"])
            )
        )
        best_score = float(
            checkpoint.get(
                "best_score",
                checkpoint["metrics"].get(
                    "best_score", checkpoint["metrics"]["total_loss"]
                ),
            )
        )
        stale = int(checkpoint.get("stale", checkpoint["metrics"].get("stale", 0)))
        resumed_from = str(args.resume)

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    bare = model.module if distributed else model
    metrics_path = args.output_dir / "metrics.csv"
    started = time.time()
    fields = None
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, lambdas, loss_scales, args
        )
        scheduler.step()
        score = train_metrics["total_loss"]
        if rank == 0:
            for split_name, metrics in (("train", train_metrics),):
                row = {"epoch": epoch, "split": split_name, **metrics}
                row = {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
                fields = fields or list(row)
                with metrics_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
            improved = score < best_score - 1e-7
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            checkpoint_extra = {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale": stale,
            }
            save_checkpoint(
                args.output_dir / "latest.pth", bare, optimizer, scheduler,
                epoch, {**train_metrics, **checkpoint_extra}, args, lambdas,
                world, initialization_report,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pth", bare, optimizer, scheduler,
                    epoch, {**train_metrics, **checkpoint_extra}, args, lambdas,
                    world, initialization_report,
                )
            print(
                f"epoch={epoch:03d} GPUs={world} train_total_loss={score:.6f} "
                f"loss_MAE={train_metrics['loss_mae']:.6f} "
                f"bpp_MAE={train_metrics['bpp_mae']:.6f} "
                f"RD_regret={train_metrics['rd_regret']:.6f} "
                f"monotonic_violation={train_metrics['bpp_monotonic_violation_rate']:.1f}",
                flush=True,
            )
        stop = torch.tensor(
            [int(stale >= args.patience if rank == 0 else 0)], device=device
        )
        if distributed:
            dist.broadcast(stop, 0)
        if stop.item():
            break

    if distributed:
        dist.barrier()
    if rank == 0:
        summary = {
            "status": "complete",
            "best_epoch": best_epoch,
            "best_training_total_loss": best_score,
            "checkpoint_selection": "minimum full-training regression total loss",
            "training_frames": len(train_set),
            "validation_frames": 0,
            "training_uses_all_merged_split_frames": True,
            "test_used_for_training_or_selection": False,
            "model_alias": "LRproxy",
            "input_feature_dim": 3,
            "optimization_targets": "six direct task-loss values + monotonic BPP; no decision head",
            "initialization": str(args.init_checkpoint),
            "resumed_from": resumed_from,
            "elapsed_seconds": time.time() - started,
            "gpus": world,
            "parameters": count_parameters(bare),
        }
        (args.output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
