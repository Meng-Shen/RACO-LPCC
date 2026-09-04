#!/usr/bin/env python3
"""Evaluate public DGCNN weights on uniformly quantized ModelNet40 objects."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def quantize_unique_resample(points: np.ndarray, step: float, count: int) -> tuple[np.ndarray, int]:
    integer = np.rint(points / step).astype(np.int32)
    integer = np.unique(integer, axis=0)
    decoded = integer.astype(np.float32) * np.float32(step)
    unique_count = len(decoded)
    if unique_count == 0:
        raise RuntimeError("Quantization produced an empty point cloud")
    if unique_count >= count:
        selected = np.linspace(0, unique_count - 1, count, dtype=np.int64)
        decoded = decoded[selected]
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
        source_index = int(self.indices[item])
        decoded, unique_count = quantize_unique_resample(
            np.asarray(self.points[source_index]), self.qstep, self.points.shape[1]
        )
        return decoded, int(self.labels[source_index]), source_index, unique_count


def load_model(source_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(source_dir))
    from model import DGCNN_cls

    args = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
    model = DGCNN_cls(args, output_channels=40)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for class_id in range(40):
        current = labels == class_id
        if current.any():
            values.append(float((predictions[current] == labels[current]).mean()))
    return float(np.mean(values))


def evaluate_level(model, points, labels, indices, qstep, args, device):
    dataset = QuantizedObjects(points, labels, indices, qstep)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    sample_indices, losses, predictions, unique_counts = [], [], [], []
    started = time.time()
    with torch.no_grad():
        for xyz, target, source_index, unique_count in loader:
            xyz = xyz.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            target_gpu = target.to(device, non_blocking=True)
            logits = model(xyz)
            loss = F.cross_entropy(logits, target_gpu, reduction="none")
            sample_indices.append(source_index.numpy())
            losses.append(loss.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            unique_counts.append(unique_count.numpy())
    sample_indices = np.concatenate(sample_indices)
    losses = np.concatenate(losses).astype(np.float32)
    predictions = np.concatenate(predictions).astype(np.int16)
    unique_counts = np.concatenate(unique_counts).astype(np.int16)
    ordered_labels = labels[sample_indices]
    return {
        "indices": sample_indices,
        "loss": losses,
        "prediction": predictions,
        "unique_count": unique_counts,
        "overall_accuracy": float((predictions == ordered_labels).mean()),
        "mean_class_accuracy": balanced_accuracy(ordered_labels, predictions),
        "mean_cross_entropy": float(losses.mean()),
        "mean_retention": float(unique_counts.mean() / points.shape[1]),
        "elapsed_seconds": time.time() - started,
    }


def parse_steps(value: str):
    steps = [float(item) for item in value.split(",") if item.strip()]
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("qsteps must contain positive values")
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--partition", choices=["train", "test"], required=True)
    parser.add_argument("--indices", default="all", help="all or path to an NPY index array")
    parser.add_argument("--qsteps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DGCNN evaluation")
    device = torch.device("cuda:0")
    data = Path(args.data_dir)
    points = np.load(data / f"{args.partition}_points.npy", mmap_mode="r")
    labels = np.load(data / f"{args.partition}_labels.npy", mmap_mode="r")
    indices = np.arange(len(points)) if args.indices == "all" else np.load(args.indices)
    qsteps = parse_steps(args.qsteps)
    model = load_model(Path(args.source_dir), Path(args.checkpoint), device)

    level_results = []
    loss_matrix = np.empty((len(indices), len(qsteps)), dtype=np.float32)
    prediction_matrix = np.empty((len(indices), len(qsteps)), dtype=np.int16)
    unique_matrix = np.empty((len(indices), len(qsteps)), dtype=np.int16)
    canonical_indices = None
    for level, qstep in enumerate(qsteps):
        result = evaluate_level(model, points, labels, indices, qstep, args, device)
        if canonical_indices is None:
            canonical_indices = result.pop("indices")
        else:
            current = result.pop("indices")
            if not np.array_equal(canonical_indices, current):
                raise RuntimeError("Sample order changed between quantization levels")
        loss_matrix[:, level] = result.pop("loss")
        prediction_matrix[:, level] = result.pop("prediction")
        unique_matrix[:, level] = result.pop("unique_count")
        result.update({"level": level, "qstep": qstep})
        level_results.append(result)
        print(json.dumps(result), flush=True)

    baseline_loss = loss_matrix[:, -1:]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=canonical_indices,
        labels=np.asarray(labels[canonical_indices], dtype=np.int16),
        qsteps=np.asarray(qsteps, dtype=np.float32),
        losses=loss_matrix,
        loss_deltas=loss_matrix - baseline_loss,
        predictions=prediction_matrix,
        unique_counts=unique_matrix,
    )
    summary = {
        "partition": args.partition,
        "samples": int(len(indices)),
        "qsteps_coarse_to_fine": qsteps,
        "baseline": f"qstep={qsteps[-1]}",
        "quantization": "round(xyz/q), unique decoded geometry, deterministic uniform repeat to 1024 points",
        "levels": level_results,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
