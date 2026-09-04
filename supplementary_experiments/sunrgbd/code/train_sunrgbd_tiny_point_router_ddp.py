#!/usr/bin/env python3
"""Train TinyPoint on 160 mm cell-mean SUN RGB-D points and six loss/BPP labels."""

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
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tiny_point_absolute_loss_monotonic_rate_proxy import (
    TinyPointAbsoluteLossMonotonicRateProxy,
    count_parameters,
)


QSTEPS_MM = (160.0, 120.0, 100.0, 80.0, 60.0, 40.0)
NUM_LEVELS = 6


def read_ids(path: Path) -> list[str]:
    return [f"{int(line):06d}" for line in path.read_text().splitlines() if line.strip()]


def load_labels(loss_csv: Path, bpp_csv: Path):
    losses = {}
    with loss_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            losses[row["scene_id"]] = np.asarray(
                [float(row[f"L{i}_total_loss"]) for i in range(NUM_LEVELS)], np.float32
            )
    rates = {}
    with bpp_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rates.setdefault(row["scene_id"], np.full(NUM_LEVELS, np.nan, np.float32))[
                int(row["rate_id"])
            ] = float(row["bpp"])
    labels = {}
    for scene_id in losses.keys() & rates.keys():
        raw_bpp = rates[scene_id]
        if np.isfinite(losses[scene_id]).all() and np.isfinite(raw_bpp).all():
            labels[scene_id] = (
                losses[scene_id], np.maximum.accumulate(raw_bpp).astype(np.float32), raw_bpp
            )
    return labels


class SUNRGBDTinyPointDataset(Dataset):
    def __init__(self, cache_dir: Path, split_file: Path, loss_csv: Path, bpp_csv: Path):
        self.scene_ids = read_ids(split_file)
        cached_ids = np.load(cache_dir / "train_scene_ids.npy", allow_pickle=False).tolist()
        if cached_ids != self.scene_ids:
            raise RuntimeError("160 mm cache scene order does not match the official train split")
        self.points = np.load(cache_dir / "train_points_160mm.npy", mmap_mode="r")
        self.offsets = np.load(cache_dir / "train_offsets_160mm.npy", mmap_mode="r")
        self.labels = load_labels(loss_csv, bpp_csv)
        missing = [scene_id for scene_id in self.scene_ids if scene_id not in self.labels]
        if missing:
            raise RuntimeError(f"Missing labels for {len(missing)} scenes: {missing[:5]}")
        all_bpp = np.stack([self.labels[scene_id][1] for scene_id in self.scene_ids])
        self.mean_log_bpp = np.log1p(all_bpp).mean(axis=0).astype(np.float32)

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        scene_id = self.scene_ids[index]
        loss, bpp, raw_bpp = self.labels[scene_id]
        return {
            "scene_id": scene_id,
            "points": torch.from_numpy(np.asarray(self.points[start:end]).copy()),
            "loss": torch.from_numpy(loss.copy()),
            "bpp": torch.from_numpy(bpp.copy()),
            "raw_bpp": torch.from_numpy(raw_bpp.copy()),
        }


def collate_variable(batch):
    return {
        "scene_ids": [item["scene_id"] for item in batch],
        "points": [item["points"] for item in batch],
        "loss": torch.stack([item["loss"] for item in batch]),
        "bpp": torch.stack([item["bpp"] for item in batch]),
        "raw_bpp": torch.stack([item["raw_bpp"] for item in batch]),
    }


def set_seed(seed: int, rank: int):
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32 - 1))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def dense_batch(point_clouds, device, training, point_cloud_range):
    """Augment, fixed-range-normalize, pad, and create a valid-point mask on CUDA."""
    lower = torch.as_tensor(point_cloud_range[:3], device=device, dtype=torch.float32)
    upper = torch.as_tensor(point_cloud_range[3:], device=device, dtype=torch.float32)
    normalized = []
    lengths = []
    for cloud in point_clouds:
        cloud = cloud.to(device, non_blocking=True).float()
        if training:
            angle = (torch.rand((), device=device) * 2.0 - 1.0) * math.pi
            cosine, sine = torch.cos(angle), torch.sin(angle)
            rotation = torch.stack([
                torch.stack([cosine, -sine, cosine.new_zeros(())]),
                torch.stack([sine, cosine, cosine.new_zeros(())]),
                torch.stack([cosine.new_zeros(()), cosine.new_zeros(()), cosine.new_ones(())]),
            ])
            cloud = cloud @ rotation.T + torch.randn_like(cloud) * 0.003
        cloud = ((cloud - lower) / (upper - lower + 1e-6)) * 2.0 - 1.0
        normalized.append(cloud)
        lengths.append(len(cloud))
    padded = pad_sequence(normalized, batch_first=True, padding_value=0.0)
    arange = torch.arange(padded.shape[1], device=device)
    mask = arange[None, :] < torch.as_tensor(lengths, device=device)[:, None]
    # Fill training padding with valid points so BatchNorm never sees artificial zeros.
    if training:
        for batch_index, length in enumerate(lengths):
            if length < padded.shape[1]:
                choices = torch.randint(length, (padded.shape[1] - length,), device=device)
                padded[batch_index, length:] = padded[batch_index, choices]
    return padded, mask, lengths


def run_epoch(model, loader, optimizer, device, loss_scales, args):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    first_batch = None
    for batch_index, batch in enumerate(loader):
        points, valid_mask, lengths = dense_batch(
            batch["points"], device, training, args.point_cloud_range
        )
        true_loss = batch["loss"].to(device, non_blocking=True)
        true_bpp = batch["bpp"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(points, valid_mask)
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
        count = len(lengths)
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
                "padded_shape": list(points.shape),
                "point_count_min": min(lengths),
                "point_count_max": max(lengths),
                "point_count_mean": float(np.mean(lengths)),
                "loss_shape": list(output["loss_pred"].shape),
                "bpp_shape": list(output["bpp_pred"].shape),
                "bpp_monotonic": bool(torch.all(torch.diff(output["bpp_pred"], dim=1) >= 0)),
            }
            print(json.dumps({"first_batch": first_batch}), flush=True)
        if args.smoke_only:
            break
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = max(float(totals[0]), 1.0)
    return {
        "samples": int(totals[0]),
        "total_loss": float(totals[1] / count),
        "loss_reg": float(totals[2] / count),
        "rate_reg": float(totals[3] / count),
        "loss_mae": float(totals[4] / count),
        "bpp_mae": float(totals[5] / count),
        "bpp_monotonic_violation_rate": float(totals[6] / (count * 5.0)),
        "mean_input_points": float(totals[7] / count),
        "first_batch": first_batch,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args,
                    loss_scales, mean_log_bpp, initialization_report):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "qsteps_mm": QSTEPS_MM,
        "loss_scales": loss_scales.detach().cpu().tolist(),
        "mean_log_bpp": mean_log_bpp.tolist(),
        "model_type": "tiny_point_160mm_six_independent_absolute_loss_plus_monotonic_bpp",
        "routing_rule": "argmin_q predicted_loss[q] + lambda * predicted_bpp[q]",
        "checkpoint_selection": "lowest full-training-set regression loss",
        "initialization_report": initialization_report,
        "parameter_counts": count_parameters(model),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--loss-csv", required=True, type=Path)
    parser.add_argument("--bpp-csv", required=True, type=Path)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-weight", type=float, default=2.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--point-cloud-range", type=float, nargs=6,
                        default=[-8.0, -8.0, -2.0, 8.0, 8.0, 6.0])
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world > 7:
        raise RuntimeError(f"Server GPU cap is seven, got {world}")
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)

    dataset = SUNRGBDTinyPointDataset(
        args.cache_dir, args.split_file, args.loss_csv, args.bpp_csv
    )
    if args.max_scenes > 0:
        dataset.scene_ids = dataset.scene_ids[:args.max_scenes]
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0 if args.smoke_only else args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_variable,
        persistent_workers=(args.workers > 0 and not args.smoke_only),
    )
    all_losses = np.stack([dataset.labels[sid][0] for sid in dataset.scene_ids])
    loss_scale_np = np.maximum(np.median(all_losses, axis=0), 1e-3).astype(np.float32)
    model = TinyPointAbsoluteLossMonotonicRateProxy(
        256, loss_scale_np, dataset.mean_log_bpp, input_channels=3
    ).to(device)
    initialization_report = model.load_compatible_heads(args.init_checkpoint)

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        target = heads if "cost_heads" in name or name.startswith("rate_head.") else backbone
        target.append(parameter)
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
            "world_size": world,
            "train_scenes": len(dataset),
            "input": "160 mm detector-aligned cells represented by original-point mean XYZ",
            "loss_scales": loss_scale_np.tolist(),
            "mean_log_bpp": dataset.mean_log_bpp.tolist(),
            "parameters": count_parameters(bare),
            "initialization": initialization_report,
        }
        (args.output_dir / "args.json").write_text(json.dumps(run_info, indent=2))
        print(json.dumps(run_info, indent=2), flush=True)

    best_loss, best_epoch, stale = math.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        metrics = run_epoch(model, loader, optimizer, device, loss_scales, args)
        scheduler.step()
        if rank == 0:
            improved = metrics["total_loss"] < best_loss - 1e-7
            if improved:
                best_loss, best_epoch, stale = metrics["total_loss"], epoch, 0
            else:
                stale += 1
            row = {"epoch": epoch, **{k: v for k, v in metrics.items() if k != "first_batch"},
                   "best_epoch": best_epoch, "best_loss": best_loss}
            metrics_path = args.output_dir / "metrics.csv"
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            save_checkpoint(args.output_dir / "latest.pth", bare, optimizer, scheduler,
                            epoch, row, args, loss_scales, dataset.mean_log_bpp,
                            initialization_report)
            if improved:
                save_checkpoint(args.output_dir / "best.pth", bare, optimizer, scheduler,
                                epoch, row, args, loss_scales, dataset.mean_log_bpp,
                                initialization_report)
            print(json.dumps(row), flush=True)
        if args.smoke_only:
            break
        stop = torch.tensor([int(stale >= args.patience if rank == 0 else 0)], device=device)
        if distributed:
            dist.broadcast(stop, 0)
        if stop.item():
            break

    if rank == 0:
        summary = {
            "status": "complete",
            "smoke_only": args.smoke_only,
            "best_epoch": best_epoch,
            "best_all_training_regression_loss": best_loss,
            "elapsed_seconds": time.time() - started,
            "gpus": world,
            "parameters": count_parameters(bare),
            "first_batch": metrics["first_batch"],
        }
        marker = "SMOKE_TEST.json" if args.smoke_only else "TRAINING_COMPLETE.json"
        (args.output_dir / marker).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
