#!/usr/bin/env python3
"""Shared label, sampling, and RD utilities for SemanticKITTI LRProxy."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def set_seed(seed: int, rank: int) -> None:
    value = int(seed + 100003 * rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def read_ids(path: Path, limit: int = 0) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return values[:limit] if limit > 0 else values


def load_semantickitti_labels(
    loss_csv: Path, bpp_csv: Path, label_cache: Path | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    if label_cache is not None:
        with np.load(label_cache) as payload:
            frame_ids = payload["frame_ids"].tolist()
            loss = payload["loss"].copy()
            bpp = payload["bpp"].copy()
            quality = payload["quality"].copy()
            num_points = payload["num_points"].copy()
        return {
            str(frame_id): (
                loss[index],
                bpp[index],
                quality[index],
                int(num_points[index]),
            )
            for index, frame_id in enumerate(frame_ids)
        }

    loss_rows = {
        row["frame_id"]: row for row in csv.DictReader(loss_csv.open(newline=""))
    }
    bpp_rows = {
        (row["scene_id"], int(row["rate_id"])): row
        for row in csv.DictReader(bpp_csv.open(newline=""))
    }
    labels = {}
    for frame_id, row in loss_rows.items():
        loss = np.asarray(
            [float(row[f"L{i}_loss_delta"]) for i in range(6)], np.float32
        )
        bpp = np.asarray(
            [float(bpp_rows[(frame_id, i)]["bpp"]) for i in range(6)],
            np.float32,
        )
        points = int(bpp_rows[(frame_id, 0)]["num_points"])
        baseline_miou = float(row["baseline_decoded_miou"])
        quality = baseline_miou - loss
        if abs(float(loss[5])) > 1e-6:
            raise ValueError(f"L5 loss is not zero for {frame_id}")
        if np.any(np.diff(bpp) < -1e-7):
            raise ValueError(f"Non-monotonic BPP labels for {frame_id}: {bpp}")
        labels[frame_id] = (loss, bpp, quality.astype(np.float32), points)
    return labels


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, rank: int, world: int) -> None:
        self.indices = list(range(rank, len(dataset), world))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def normalized_curve_auc(bpp, quality) -> float:
    bpp = np.asarray(bpp, np.float64)
    quality = np.asarray(quality, np.float64)
    order = np.argsort(bpp)
    x = np.log(np.maximum(bpp[order], 1e-9))
    y = quality[order]
    if len(x) < 2 or x[-1] - x[0] < 1e-12:
        return float(y.mean())
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def rd_levels(
    loss_by_level: torch.Tensor,
    bpp_by_level: torch.Tensor,
    lambdas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``argmin_q loss(q) + lambda * BPP(q)`` for every sample."""
    scores = (
        loss_by_level[:, None, :]
        + lambdas[None, :, None] * bpp_by_level[:, None, :]
    )
    return scores.argmin(dim=-1), scores


__all__ = [
    "DistributedEvalSampler",
    "load_semantickitti_labels",
    "normalized_curve_auc",
    "rd_levels",
    "read_ids",
    "set_seed",
]
