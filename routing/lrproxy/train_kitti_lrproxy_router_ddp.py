#!/usr/bin/env python3
"""Train three-coordinate LRproxy on complete KITTI FOV labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from gpu_voxelizer import voxelize_batch_gpu
from lrproxy import (
    LRProxy,
    count_parameters,
    select_xyz_features,
)
from kitti_lrproxy_training_utils import (
    KITTIRouterDataset,
    calibrate_lambdas,
    collate_raw,
    move_and_augment,
    training_scales,
)


NUM_LEVELS = 6
QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)


def set_seed(seed: int, rank: int):
    value = int(seed + rank * 100003)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def pack_voxel_features(features, coords, batch_size: int, training: bool):
    """Convert flattened spconv-style voxel features to masked dense point batches."""
    clouds = [features[coords[:, 0].long() == index] for index in range(batch_size)]
    lengths = [int(cloud.shape[0]) for cloud in clouds]
    if min(lengths) <= 0:
        raise RuntimeError(f"Empty LRproxy cloud in batch: {lengths}")
    padded = pad_sequence(clouds, batch_first=True, padding_value=0.0)
    sequence = torch.arange(padded.shape[1], device=features.device)
    valid_mask = sequence[None, :] < torch.as_tensor(lengths, device=features.device)[:, None]
    # Preserve the existing LRProxy BatchNorm behavior without artificial zero samples.
    if training:
        for batch_index, length in enumerate(lengths):
            if length < padded.shape[1]:
                choices = torch.randint(length, (padded.shape[1] - length,), device=features.device)
                padded[batch_index, length:] = padded[batch_index, choices]
    return padded, valid_mask, lengths


def run_epoch(model, loader, optimizer, device, loss_scales, args):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    first_batch = None
    for batch_index, batch in enumerate(loader):
        true_loss = batch["loss_by_level"].to(device, non_blocking=True)
        true_bpp = batch["bpp_by_level"].to(device, non_blocking=True)
        point_clouds = move_and_augment(
            batch["points"], device, training, args.jitter_std, args.rotation_aug
        )
        voxel_features, voxel_coords = voxelize_batch_gpu(
            point_clouds,
            args.voxel_size,
            args.point_cloud_range,
            args.max_voxels,
            use_abs_xyz=True,
            include_intensity=False,
            random_subsample=training,
        )
        voxel_features = select_xyz_features(voxel_features)
        packed, valid_mask, lengths = pack_voxel_features(
            voxel_features, voxel_coords, len(point_clouds), training
        )
        if packed.shape[-1] != 3:
            raise RuntimeError(f"LRproxy expected 3 features, got {packed.shape[-1]}")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(packed, valid_mask)
            loss_reg = F.smooth_l1_loss(
                output["loss_pred"] / loss_scales[None, :],
                true_loss / loss_scales[None, :],
            )
            rate_reg = F.smooth_l1_loss(output["rate_log_pred"], torch.log1p(true_bpp))
            total = args.loss_weight * loss_reg + args.rate_weight * rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        count = len(point_clouds)
        totals[0] += count
        totals[1] += float(total.detach()) * count
        totals[2] += float(loss_reg.detach()) * count
        totals[3] += float(rate_reg.detach()) * count
        totals[4] += float(torch.abs(output["loss_pred"] - true_loss).mean()) * count
        totals[5] += float(torch.abs(output["bpp_pred"] - true_bpp).mean()) * count
        totals[6] += float((torch.diff(output["bpp_pred"], dim=1) < 0).sum())
        totals[7] += float(sum(lengths))
        if batch_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            first_batch = {
                "batch_size": count,
                "raw_points": int(sum(len(cloud) for cloud in point_clouds)),
                "active_voxels": int(sum(lengths)),
                "voxel_count_min": min(lengths),
                "voxel_count_max": max(lengths),
                "padded_shape": list(packed.shape),
                "feature_dim": int(packed.shape[-1]),
                "loss_shape": list(output["loss_pred"].shape),
                "bpp_shape": list(output["bpp_pred"].shape),
                "bpp_monotonic": bool(torch.all(torch.diff(output["bpp_pred"], dim=1) >= 0)),
            }
            print(json.dumps({"first_batch": first_batch}), flush=True)
    if dist.is_initialized():
        dist.all_reduce(totals)
    samples = max(float(totals[0]), 1.0)
    return {
        "samples": int(totals[0]),
        "total_loss": float(totals[1] / samples),
        "loss_regression_normalized": float(totals[2] / samples),
        "rate_regression_log1p": float(totals[3] / samples),
        "loss_mae": float(totals[4] / samples),
        "bpp_mae": float(totals[5] / samples),
        "bpp_monotonic_violation_rate": float(totals[6] / (samples * 5.0)),
        "mean_active_voxels": float(totals[7] / samples),
        "first_batch": first_batch,
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, metrics, args, lambdas,
                       mean_log_bpp, loss_scales, initialization, world, best_epoch,
                       best_train_loss):
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "lambdas": lambdas.detach().cpu().tolist(),
        "qsteps_mm": list(QSTEPS_MM),
        "loss_scales": loss_scales.detach().cpu().tolist(),
        "mean_log_bpp": mean_log_bpp.tolist(),
        "model_alias": "LRproxy",
        "model_type": "lrproxy_six_absolute_pvrcnn_loss_plus_monotonic_bpp",
        "input_feature_semantics": "normalized voxel-mean absolute XYZ (3)",
        "input_feature_dim": 3,
        "routing_rule": "argmin_q predicted_absolute_loss(q) + lambda * predicted_BPP(q)",
        "selection_metric": "lowest joint regression loss on complete official KITTI train",
        "best_epoch": int(best_epoch),
        "best_train_total_loss": float(best_train_loss),
        "former_holdout_included_in_training": True,
        "official_kitti_val_used_for_selection": False,
        "world_size": int(world),
        "initialization_report": initialization,
        "parameter_counts": count_parameters(model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-dir", required=True, type=Path)
    parser.add_argument("--loss-csv", required=True, type=Path)
    parser.add_argument("--bpp-csv", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-weight", type=float, default=2.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.16, 0.16, 0.16])
    parser.add_argument("--point-cloud-range", type=float, nargs=6,
                        default=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    parser.add_argument("--max-voxels", type=int, default=50000)
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--rotation-aug", action="store_true")
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
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
    set_seed(args.seed, rank)

    dataset = KITTIRouterDataset(
        args.points_dir, args.train_split, args.loss_csv, args.bpp_csv,
        args.max_train_frames,
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers, pin_memory=True,
        drop_last=False, collate_fn=collate_raw,
        persistent_workers=args.workers > 0,
    )
    loss_scale_np, mean_log_bpp = training_scales(dataset)
    lambda_payload = calibrate_lambdas(dataset)
    lambda_payload["source"] = "complete official KITTI train including former holdout"
    lambda_payload["former_holdout_included_in_training"] = True
    lambdas = torch.tensor(
        lambda_payload["lambdas_low_rate_to_high_rate"], dtype=torch.float32, device=device
    )
    model = LRProxy(
        256, loss_scale_np, mean_log_bpp
    ).to(device)
    initialization = model.load_full_checkpoint(args.init_checkpoint)
    if initialization["new_backbone_randomly_initialized"]:
        raise RuntimeError(f"LRProxy initialization was incomplete: {initialization}")
    parameters = count_parameters(model)
    if parameters["total"] != 540460:
        raise RuntimeError(f"Unexpected LRproxy parameter count: {parameters}")

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if "cost_heads" in name or name.startswith("rate_head.") else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    bare = model.module if distributed else model
    loss_scales = torch.tensor(loss_scale_np, dtype=torch.float32, device=device)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_info = {
            "args": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
            "model_alias": "LRproxy",
            "input_feature_dim": 3,
            "input_feature_semantics": "normalized voxel-mean absolute XYZ (3)",
            "world_size": world,
            "unique_train_frames": len(dataset),
            "former_holdout_included_in_training": True,
            "official_kitti_val_used_for_selection": False,
            "parameters": parameters,
            "initialization": initialization,
            "lambdas": lambda_payload,
        }
        (args.output_dir / "args.json").write_text(json.dumps(run_info, indent=2))
        (args.output_dir / "initialization_report.json").write_text(
            json.dumps(initialization, indent=2)
        )
        (args.output_dir / "lambda_calibration_train_only.json").write_text(
            json.dumps(lambda_payload, indent=2)
        )
        print(json.dumps(run_info, indent=2), flush=True)

    metrics_csv = args.output_dir / "metrics.csv"
    best_epoch, best_train_loss = 0, math.inf
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_started = time.time()
        metrics = run_epoch(model, loader, optimizer, device, loss_scales, args)
        scheduler.step()
        if rank == 0:
            improved = metrics["total_loss"] < best_train_loss - 1e-9
            if improved:
                best_epoch, best_train_loss = epoch, metrics["total_loss"]
            row = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - started,
                "epoch_seconds": time.time() - epoch_started,
                **{key: value for key, value in metrics.items() if key != "first_batch"},
                "best_epoch": best_epoch,
                "best_train_total_loss": best_train_loss,
            }
            payload = checkpoint_payload(
                bare, optimizer, scheduler, epoch, row, args, lambdas,
                mean_log_bpp, loss_scales, initialization, world,
                best_epoch, best_train_loss,
            )
            torch.save(payload, args.output_dir / "latest.pth")
            if improved:
                torch.save(payload, args.output_dir / "best.pth")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(payload, args.output_dir / f"epoch_{epoch:03d}.pth")
            with metrics_csv.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(json.dumps(row), flush=True)
        if distributed:
            dist.barrier()

    if rank == 0:
        summary = {
            "status": "training_complete",
            "model_alias": "LRproxy",
            "input_feature_dim": 3,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_train_total_loss": best_train_loss,
            "elapsed_seconds": time.time() - started,
            "gpus": world,
            "unique_train_frames": len(dataset),
            "distributed_samples_per_epoch": metrics["samples"],
            "former_holdout_included_in_training": True,
            "official_kitti_val_used_for_training_or_selection": False,
            "bpp_monotonic_violation_rate": metrics["bpp_monotonic_violation_rate"],
            "parameters": parameters,
            "first_batch": metrics["first_batch"],
        }
        (args.output_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
