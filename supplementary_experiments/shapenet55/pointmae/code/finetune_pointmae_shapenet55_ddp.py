#!/usr/bin/env python3
"""Train a 55-way head on the official ShapeNet55-pretrained Point-MAE."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from pointmae_classifier import PointMAEClassifier, load_shapenet_pretrain


class ShapeObjects(Dataset):
    def __init__(self, points_path, labels_path, indices_path, training=False):
        self.points = np.load(points_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.indices = np.load(indices_path).astype(np.int64)
        self.training = bool(training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        xyz = np.asarray(self.points[index], dtype=np.float32).copy()
        if self.training:
            np.random.shuffle(xyz)
            xyz *= np.float32(np.random.uniform(0.8, 1.25))
            xyz += np.random.uniform(-0.10, 0.10, size=(1, 3)).astype(np.float32)
        return torch.from_numpy(xyz), int(self.labels[index]), index


def seed_everything(seed, rank):
    value = int(seed + rank * 100003)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def confusion_metrics(confusion: torch.Tensor):
    confusion = confusion.double()
    total = confusion.sum().item()
    support = confusion.sum(dim=1)
    recalls = confusion.diag()[support > 0] / support[support > 0]
    return {
        "overall_accuracy": float(confusion.diag().sum().item() / max(total, 1.0)),
        "mean_class_accuracy": float(recalls.mean()) if len(recalls) else 0.0,
        "samples": int(total),
    }


def run_epoch(model, loader, device, classes, optimizer=None, scaler=None, label_smoothing=0.0):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    count = torch.zeros(1, dtype=torch.float64, device=device)
    confusion = torch.zeros((classes, classes), dtype=torch.int64, device=device)
    for xyz, labels, _indices in loader:
        xyz = xyz.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(xyz)
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
        prediction = logits.argmax(dim=1)
        confusion += torch.bincount(
            labels * classes + prediction, minlength=classes * classes
        ).reshape(classes, classes)
        loss_sum += loss.detach().double() * len(labels)
        count += len(labels)
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
        dist.all_reduce(confusion)
    result = confusion_metrics(confusion)
    result["cross_entropy"] = float(loss_sum / count.clamp_min(1))
    return result


def make_loader(dataset, batch_size, workers, sampler, training):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and training,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", type=int, default=55)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--workers", type=int, default=4, help="Per-GPU workers")
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--train-backbone", action="store_true",
        help="Also fine-tune the pretrained encoder; default freezes it and trains only the 55-way head",
    )
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
    seed_everything(args.seed, rank)

    data = Path(args.data_dir).resolve()
    output = Path(args.output_dir).resolve()
    pretrained = Path(args.pretrained).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    train_set = ShapeObjects(
        data / "all_points.npy", data / "labels.npy", data / "model_train_indices.npy", True
    )
    val_set = ShapeObjects(
        data / "all_points.npy", data / "labels.npy", data / "model_val_indices.npy", False
    )
    test_set = ShapeObjects(
        data / "all_points.npy", data / "labels.npy", data / "test_indices.npy", False
    )
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None
    train_loader = make_loader(train_set, args.batch_size, args.workers, train_sampler, True)
    val_loader = make_loader(val_set, args.batch_size, args.workers, val_sampler, False)
    test_loader = make_loader(test_set, args.batch_size, args.workers, test_sampler, False)

    model = PointMAEClassifier(num_classes=args.classes)
    load_report = load_shapenet_pretrain(model, pretrained)
    head_prefixes = ("cls_head_finetune.", "cls_token", "cls_pos")
    backbone = [p for name, p in model.named_parameters() if not name.startswith(head_prefixes)]
    head = [p for name, p in model.named_parameters() if name.startswith(head_prefixes)]
    if not args.train_backbone:
        for parameter in backbone:
            parameter.requires_grad_(False)
        parameter_groups = [{"params": head, "lr": args.head_lr, "name": "head"}]
    else:
        parameter_groups = [
            {"params": backbone, "lr": args.backbone_lr, "name": "backbone"},
            {"params": head, "lr": args.head_lr, "name": "head"},
        ]
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True)
    bare = model.module if distributed else model
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    metrics_path = output / "metrics.csv"
    best_score, best_epoch, stale = -math.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, args.classes, optimizer, scaler, args.label_smoothing
        )
        val_metrics = run_epoch(model, val_loader, device, args.classes)
        scheduler.step()
        stop = False
        if rank == 0:
            fields = [
                "epoch", "split", "overall_accuracy", "mean_class_accuracy",
                "cross_entropy", "samples", "backbone_lr", "head_lr",
            ]
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if handle.tell() == 0:
                    writer.writeheader()
                for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                    current_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
                    writer.writerow({
                        "epoch": epoch, "split": split, **metrics,
                        "backbone_lr": current_lrs.get("backbone", 0.0),
                        "head_lr": current_lrs["head"],
                    })
            score = val_metrics["overall_accuracy"]
            checkpoint = {
                "epoch": epoch,
                "model": bare.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "validation": val_metrics,
                "args": vars(args),
                "classes": args.classes,
                "selection_metric": "validation overall accuracy",
                "pretraining_dataset": "ShapeNet55",
                "pretraining_task": "Point-MAE self-supervised masked point modeling",
                "pretrained": str(pretrained),
                "pretrained_sha256": sha256(pretrained),
                "pretrain_load_report": load_report,
            }
            torch.save(checkpoint, output / "latest.pth")
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save(checkpoint, output / "best.pth")
            else:
                stale += 1
            print(
                f"epoch={epoch:03d} train_acc={train_metrics['overall_accuracy']:.5f} "
                f"val_acc={val_metrics['overall_accuracy']:.5f} "
                f"val_macc={val_metrics['mean_class_accuracy']:.5f} stale={stale}",
                flush=True,
            )
            stop = stale >= args.patience
        flag = torch.tensor([int(stop)], device=device)
        if distributed:
            dist.broadcast(flag, src=0)
        if flag.item():
            break

    if distributed:
        dist.barrier()
    best = torch.load(output / "best.pth", map_location=device)
    bare.load_state_dict(best["model"])
    test_metrics = run_epoch(model, test_loader, device, args.classes)
    if rank == 0:
        summary = {
            "classifier": "Point-MAE, XYZ only, 1024 points, 55 classes",
            "pretraining": "official Point-MAE self-supervised checkpoint trained on ShapeNet55",
            "fine_tuning": (
                "new 55-way head and the ShapeNet55-pretrained backbone"
                if args.train_backbone else
                "frozen ShapeNet55-pretrained backbone; only the new 55-way head is trained"
            ),
            "gpus": world,
            "best_epoch": best_epoch,
            "best_validation_overall_accuracy": best_score,
            "best_validation_metrics": best["validation"],
            "official_test_metrics": test_metrics,
            "test_used_for_checkpoint_selection": False,
            "elapsed_seconds": time.time() - started,
        }
        (output / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
