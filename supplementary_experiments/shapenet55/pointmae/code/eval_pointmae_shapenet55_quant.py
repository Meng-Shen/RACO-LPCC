#!/usr/bin/env python3
"""Extract per-object Point-MAE loss/predictions at ShapeNet55 quantization levels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pointmae_classifier import PointMAEClassifier


def quantize_unique_resample(points: np.ndarray, step: float, count: int):
    integer = np.unique(np.rint(points / step).astype(np.int32), axis=0)
    decoded = integer.astype(np.float32) * np.float32(step)
    unique_count = len(decoded)
    if unique_count == 0:
        raise RuntimeError("Quantization produced an empty point cloud")
    if unique_count >= count:
        choice = np.linspace(0, unique_count - 1, count, dtype=np.int64)
        decoded = decoded[choice]
    else:
        repeats, remainder = divmod(count, unique_count)
        decoded = np.concatenate([np.tile(decoded, (repeats, 1)), decoded[:remainder]], axis=0)
    return decoded.astype(np.float32, copy=False), unique_count


class QuantizedObjects(Dataset):
    def __init__(self, points, labels, indices, qstep):
        self.points = points
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.qstep = float(qstep)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        decoded, unique_count = quantize_unique_resample(
            np.asarray(self.points[index]), self.qstep, self.points.shape[1]
        )
        return decoded, int(self.labels[index]), index, unique_count


def parse_steps(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("qsteps must be positive")
    return values


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = int(checkpoint.get("classes", 55))
    model = PointMAEClassifier(num_classes=classes)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    clean = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model.load_state_dict(clean, strict=True)
    return model.to(device).eval(), classes


def metrics(labels, predictions, classes):
    recalls = []
    for class_id in range(classes):
        mask = labels == class_id
        if mask.any():
            recalls.append(float((predictions[mask] == labels[mask]).mean()))
    return float((predictions == labels).mean()), float(np.mean(recalls))


def evaluate(model, points, labels, indices, qstep, batch_size, workers, device, classes):
    loader = DataLoader(
        QuantizedObjects(points, labels, indices, qstep),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    all_indices, losses, predictions, unique_counts = [], [], [], []
    started = time.time()
    with torch.no_grad():
        for xyz, target, source_index, unique_count in loader:
            xyz = xyz.to(device, non_blocking=True).contiguous()
            target_gpu = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(xyz)
            losses.append(F.cross_entropy(logits.float(), target_gpu, reduction="none").cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_indices.append(source_index.numpy())
            unique_counts.append(unique_count.numpy())
    all_indices = np.concatenate(all_indices).astype(np.int64)
    losses = np.concatenate(losses).astype(np.float32)
    predictions = np.concatenate(predictions).astype(np.int16)
    unique_counts = np.concatenate(unique_counts).astype(np.int16)
    ordered_labels = np.asarray(labels[all_indices], dtype=np.int64)
    overall, mean_class = metrics(ordered_labels, predictions, classes)
    return all_indices, losses, predictions, unique_counts, {
        "qstep": qstep,
        "samples": len(all_indices),
        "overall_accuracy": overall,
        "mean_class_accuracy": mean_class,
        "mean_cross_entropy": float(losses.mean()),
        "mean_retention": float(unique_counts.mean() / points.shape[1]),
        "elapsed_seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--qsteps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    data = Path(args.data_dir).resolve()
    points = np.load(data / "all_points.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    indices = np.load(args.indices).astype(np.int64)
    qsteps = parse_steps(args.qsteps)
    model, classes = load_model(Path(args.checkpoint), device)

    losses = np.empty((len(indices), len(qsteps)), dtype=np.float32)
    predictions = np.empty((len(indices), len(qsteps)), dtype=np.int16)
    unique_counts = np.empty((len(indices), len(qsteps)), dtype=np.int16)
    canonical = None
    levels = []
    for level, qstep in enumerate(qsteps):
        current, level_loss, level_prediction, level_unique, summary = evaluate(
            model, points, labels, indices, qstep,
            args.batch_size, args.workers, device, classes,
        )
        if canonical is None:
            canonical = current
        elif not np.array_equal(canonical, current):
            raise RuntimeError("Sample order changed between quantization levels")
        losses[:, level] = level_loss
        predictions[:, level] = level_prediction
        unique_counts[:, level] = level_unique
        summary["level"] = level
        levels.append(summary)
        print(json.dumps(summary), flush=True)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=canonical,
        labels=np.asarray(labels[canonical], dtype=np.int16),
        qsteps=np.asarray(qsteps, dtype=np.float32),
        losses=losses,
        loss_deltas=losses - losses[:, -1:],
        predictions=predictions,
        unique_counts=unique_counts,
    )
    manifest = {
        "samples": int(len(indices)),
        "classes": classes,
        "model": "Point-MAE fine-tuned from the official ShapeNet55 self-supervised checkpoint",
        "qsteps_coarse_to_fine": qsteps,
        "reference_rate": f"qstep={qsteps[-1]}",
        "quantization": "round(XYZ/q), merge duplicates, deterministic repeat to 1024 points",
        "levels": levels,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
