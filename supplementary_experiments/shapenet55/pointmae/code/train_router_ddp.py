#!/usr/bin/env python3
"""Three-GPU DDP training for the ShapeNet55 loss+BPP routing proxy."""

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
from torch.utils.data import DataLoader, DistributedSampler

from train_router import (
    RateAwareProxy,
    RoutingObjects,
    choose_levels,
    collate,
    curve_auc,
    export_test,
    losses_by_level,
)


def set_seed(seed, rank):
    value = int(seed + 100003 * rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def run_epoch(model, loader, optimizer, device, lambdas, scale):
    training = optimizer is not None
    model.train(training)
    count = 0.0
    total_sum = loss_sum = rate_sum = loss_abs = rate_abs = regret_sum = 0.0
    correct_levels = np.zeros(len(lambdas), dtype=np.float64)
    chosen_correct = np.zeros(len(lambdas), dtype=np.float64)
    chosen_bpp = np.zeros(len(lambdas), dtype=np.float64)
    for batch in loader:
        for key in ("head_target", "loss", "bpp", "correct"):
            batch[key] = batch[key].to(device, non_blocking=True)
        features = batch["voxel_features"].to(device, non_blocking=True)
        coords = batch["voxel_coords"].to(device, non_blocking=True)
        size = int(batch["batch_size"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            cost_pred, rate_log, bpp_pred = model(features, coords, size)
            loss_reg = F.smooth_l1_loss(cost_pred, batch["head_target"])
            rate_reg = F.smooth_l1_loss(rate_log, torch.log1p(batch["bpp"]))
            total = loss_reg + rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        with torch.no_grad():
            predicted_loss = losses_by_level(cost_pred, scale)
            predicted_levels, _ = choose_levels(predicted_loss, bpp_pred, lambdas)
            oracle_levels, true_scores = choose_levels(batch["loss"], batch["bpp"], lambdas)
            selected_scores = torch.gather(true_scores, 2, predicted_levels[:, :, None]).squeeze(-1)
            optimal_scores = true_scores.min(dim=-1).values
            true_selected_bpp = torch.gather(batch["bpp"], 1, predicted_levels)
            selected_correct = torch.gather(batch["correct"], 1, predicted_levels)
            count += size
            total_sum += float(total) * size
            loss_sum += float(loss_reg) * size
            rate_sum += float(rate_reg) * size
            loss_abs += float(torch.abs(predicted_loss - batch["loss"]).mean()) * size
            rate_abs += float(torch.abs(bpp_pred - batch["bpp"]).mean()) * size
            regret_sum += float((selected_scores - optimal_scores).mean()) * size
            correct_levels += (predicted_levels == oracle_levels).sum(dim=0).cpu().numpy()
            chosen_correct += selected_correct.sum(dim=0).cpu().numpy()
            chosen_bpp += true_selected_bpp.sum(dim=0).cpu().numpy()

    packed = torch.tensor(
        [count, total_sum, loss_sum, rate_sum, loss_abs, rate_abs, regret_sum]
        + correct_levels.tolist() + chosen_correct.tolist() + chosen_bpp.tolist(),
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().numpy()
    count, total_sum, loss_sum, rate_sum, loss_abs, rate_abs, regret_sum = values[:7]
    offset = 7
    correct_levels = values[offset:offset + len(lambdas)]
    offset += len(lambdas)
    chosen_correct = values[offset:offset + len(lambdas)]
    offset += len(lambdas)
    chosen_bpp = values[offset:offset + len(lambdas)]
    accuracy = chosen_correct / max(count, 1.0)
    mean_bpp = chosen_bpp / max(count, 1.0)
    return {
        "samples": int(count),
        "total_loss": total_sum / count,
        "loss_regression": loss_sum / count,
        "rate_regression": rate_sum / count,
        "loss_mae": loss_abs / count,
        "bpp_mae": rate_abs / count,
        "rd_regret": regret_sum / count,
        "selection_accuracy": (correct_levels / count).tolist(),
        "mean_selection_accuracy": float(correct_levels.mean() / count),
        "curve_accuracy": accuracy.tolist(),
        "curve_bpp": mean_bpp.tolist(),
        "accuracy_bpp_auc": curve_auc(mean_bpp, accuracy),
    }


def make_loader(dataset, batch_size, workers, training, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=training and sampler is None,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=collate,
        persistent_workers=workers > 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--bpp", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--val-indices", required=True)
    parser.add_argument("--test-indices", required=True)
    parser.add_argument("--lambda-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--workers", type=int, default=4, help="Per-GPU data workers")
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--max-voxels", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 3:
        raise RuntimeError(f"This task is capped at three GPUs, got WORLD_SIZE={world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)

    output = Path(args.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    quant_train = np.load(args.quant)
    p95 = max(float(np.quantile(np.abs(quant_train["loss_deltas"][:, :5]), 0.95)), 1e-4)
    target_scale = min(100.0, max(0.1, 0.5 / p95))
    train_indices = np.load(args.train_indices)
    val_indices = np.load(args.val_indices)
    test_indices = np.load(args.test_indices)
    common = dict(
        target_scale=target_scale,
        voxel_size=[args.voxel_size] * 3,
        point_cloud_range=[-1.1, -1.1, -1.1, 1.1, 1.1, 1.1],
        max_voxels=args.max_voxels,
    )
    train_set = RoutingObjects(args.points, args.quant, args.bpp, train_indices, **common)
    val_set = RoutingObjects(args.points, args.quant, args.bpp, val_indices, **common)
    test_set = RoutingObjects(args.points, args.quant, args.bpp, test_indices, **common)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = make_loader(train_set, args.batch_size, args.workers, True, train_sampler)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False, val_sampler)

    lambda_data = json.loads(Path(args.lambda_json).read_text())
    lambdas = torch.tensor(
        lambda_data["lambdas_high_rate_to_low_rate"], dtype=torch.float32, device=device
    )
    model = RateAwareProxy(train_set.spatial_shape, args.feat_dim, train_set.mean_log_bpp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_score, best_epoch, stale = -math.inf, 0, 0
    resumed_from = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        checkpoint = torch.load(resume_path, map_location=device)
        state = checkpoint["model"]
        state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("metrics", {}).get("accuracy_bpp_auc", -math.inf))
        resumed_from = str(resume_path)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True)
    bare = model.module if distributed else model

    metrics_path = output / "metrics.csv"
    fields = None
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, device, lambdas, target_scale)
        val_metrics = run_epoch(model, val_loader, None, device, lambdas, target_scale)
        scheduler.step()
        score = val_metrics["accuracy_bpp_auc"]
        if rank == 0:
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                row = {"epoch": epoch, "split": split, **metrics}
                row = {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
                if fields is None:
                    fields = list(row)
                with metrics_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
            checkpoint = {
                "epoch": epoch,
                "model": bare.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": val_metrics,
                "args": vars(args),
                "target_scale": target_scale,
                "lambdas": lambdas.cpu().tolist(),
                "qsteps": train_set.qsteps.tolist(),
                "selection_metric": "validation Accuracy-BPP AUC",
                "world_size": world,
            }
            torch.save(checkpoint, output / "latest.pth")
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save(checkpoint, output / "best.pth")
            else:
                stale += 1
            print(
                f"epoch={epoch:03d} GPUs={world} val_auc={score:.6f} "
                f"val_loss_mae={val_metrics['loss_mae']:.5f} "
                f"val_bpp_mae={val_metrics['bpp_mae']:.5f} "
                f"selection={val_metrics['mean_selection_accuracy']:.4f}",
                flush=True,
            )
        stop = torch.tensor([int(stale >= args.patience if rank == 0 else 0)], device=device)
        if distributed:
            dist.broadcast(stop, src=0)
        if stop.item():
            break

    if distributed:
        dist.barrier()
    best = torch.load(output / "best.pth", map_location=device)
    bare.load_state_dict(best["model"])
    if rank == 0:
        test_loader = make_loader(test_set, args.batch_size, args.workers, False)
        export_test(
            bare, test_loader, device, lambdas, target_scale, test_set.qsteps,
            output / "test_router_predictions.npz",
        )
        summary = {
            "dataset": "ShapeNet55 official split; test held out",
            "best_epoch": best_epoch,
            "best_validation_accuracy_bpp_auc": best_score,
            "best_validation_metrics": best["metrics"],
            "elapsed_seconds_this_run": time.time() - started,
            "model_type": "shared sparse XYZ backbone + five CE-loss heads + one six-rate BPP head",
            "optimization_targets": "loss regression + BPP regression only",
            "checkpoint_selection": "validation Accuracy-BPP AUC",
            "test_used_for_checkpoint_selection": False,
            "target_scale": target_scale,
            "gpus": world,
            "resumed_from": resumed_from,
        }
        (output / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
