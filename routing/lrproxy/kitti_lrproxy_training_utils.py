#!/usr/bin/env python3
"""Shared KITTI data and calibration utilities for LRProxy routers."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_LEVELS = 6


def read_ids(path: Path, limit: int = 0) -> list[str]:
    values = [
        line.strip().zfill(6)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return values[:limit] if limit > 0 else values


def load_labels(
    loss_csv: Path, bpp_csv: Path
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    loss_rows = {
        str(row["frame_id"]).zfill(6): row
        for row in csv.DictReader(loss_csv.open(newline=""))
    }
    bpp_rows = {}
    for row in csv.DictReader(bpp_csv.open(newline="")):
        frame_id = str(row.get("filename", row.get("frame_id", ""))).zfill(6)
        bpp_rows[(frame_id, int(row["rate_id"]))] = row

    labels = {}
    for frame_id, row in loss_rows.items():
        losses = np.asarray(
            [float(row[f"L{i}_total_loss"]) for i in range(NUM_LEVELS)],
            np.float32,
        )
        try:
            rates = np.asarray(
                [float(bpp_rows[(frame_id, i)]["bpp"]) for i in range(NUM_LEVELS)],
                np.float32,
            )
            points = int(bpp_rows[(frame_id, 0)]["num_points"])
        except KeyError:
            continue
        if np.any(~np.isfinite(losses)) or np.any(losses < 0):
            raise ValueError(f"Invalid absolute PV-RCNN losses for {frame_id}: {losses}")
        if np.any(np.diff(rates) < -1e-7):
            raise ValueError(f"Non-monotonic G-PCC BPP for {frame_id}: {rates}")
        labels[frame_id] = (losses, rates, points)
    return labels


class KITTIRouterDataset(Dataset):
    def __init__(
        self,
        points_dir: Path,
        split: Path,
        loss_csv: Path,
        bpp_csv: Path,
        limit: int = 0,
    ) -> None:
        self.points_dir = points_dir
        self.ids = read_ids(split, limit)
        self.labels = load_labels(loss_csv, bpp_csv)
        for frame_id in self.ids:
            if frame_id not in self.labels:
                raise KeyError(f"Missing six loss/BPP labels for {frame_id}")
            path = self.points_dir / f"{frame_id}.bin"
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        frame_id = self.ids[index]
        raw = np.fromfile(self.points_dir / f"{frame_id}.bin", dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"Invalid KITTI point cloud: {frame_id}")
        points = torch.from_numpy(raw.reshape(-1, 4)[:, :3].copy())
        losses, rates, point_count = self.labels[frame_id]
        return {
            "frame_id": frame_id,
            "points": points,
            "loss_by_level": torch.from_numpy(losses.copy()),
            "bpp_by_level": torch.from_numpy(rates.copy()),
            "num_points": int(point_count),
        }


def collate_raw(batch: list[dict]) -> dict:
    return {
        "frame_id": [item["frame_id"] for item in batch],
        "points": [item["points"] for item in batch],
        "loss_by_level": torch.stack([item["loss_by_level"] for item in batch]),
        "bpp_by_level": torch.stack([item["bpp_by_level"] for item in batch]),
        "num_points": torch.tensor(
            [item["num_points"] for item in batch], dtype=torch.float64
        ),
    }


def training_scales(dataset: KITTIRouterDataset) -> tuple[np.ndarray, np.ndarray]:
    losses = np.stack([dataset.labels[value][0] for value in dataset.ids])
    rates = np.stack([dataset.labels[value][1] for value in dataset.ids])
    loss_scales = np.maximum(np.median(losses, axis=0), np.float32(1e-3))
    mean_log_bpp = np.log1p(rates).mean(axis=0).astype(np.float32)
    return loss_scales.astype(np.float32), mean_log_bpp


def calibrate_lambdas(dataset: KITTIRouterDataset, count: int = 6) -> dict:
    losses = torch.from_numpy(
        np.stack([dataset.labels[value][0] for value in dataset.ids])
    ).double()
    bpp = torch.from_numpy(
        np.stack([dataset.labels[value][1] for value in dataset.ids])
    ).double()
    points = torch.tensor(
        [dataset.labels[value][2] for value in dataset.ids], dtype=torch.double
    )
    candidate_lambdas = torch.cat(
        [torch.logspace(3, -5, 801, dtype=torch.double), torch.zeros(1, dtype=torch.double)]
    )
    aggregate_rates = []
    mean_levels = []
    for value in candidate_lambdas:
        selected = torch.argmin(losses + value * bpp, dim=1)
        chosen = bpp.gather(1, selected[:, None]).squeeze(1)
        aggregate_rates.append(float((chosen * points).sum() / points.sum()))
        mean_levels.append(float(selected.double().mean()))
    aggregate_rates_np = np.asarray(aggregate_rates)
    lo = max(float(aggregate_rates_np.min()), 1e-9)
    hi = max(float(aggregate_rates_np.max()), lo * (1.0 + 1e-9))
    targets = np.exp(np.linspace(math.log(lo), math.log(hi), count))
    picked = []
    used = set()
    for target in targets:
        order = np.argsort(
            np.abs(np.log(np.maximum(aggregate_rates_np, 1e-9)) - math.log(target))
        )
        index = next(
            (int(item) for item in order if int(item) not in used), int(order[0])
        )
        used.add(index)
        picked.append(index)
    picked.sort(key=lambda index: aggregate_rates_np[index])
    return {
        "source": "official KITTI train subset only; no official val/test frames",
        "ordering": "low-rate to high-rate",
        "lambdas_low_rate_to_high_rate": [
            float(candidate_lambdas[index]) for index in picked
        ],
        "calibrated_total_bits_over_total_points_bpp": [
            float(aggregate_rates_np[index]) for index in picked
        ],
        "mean_selected_level_coarse_to_fine": [
            float(mean_levels[index]) for index in picked
        ],
        "target_bpp": targets.tolist(),
    }


def move_and_augment(points, device, training, jitter_std, rotation_aug):
    moved = []
    for frame in points:
        frame = frame.to(device, non_blocking=True)
        if training and rotation_aug:
            angle = torch.empty((), device=device).uniform_(-math.pi, math.pi)
            cosine, sine = torch.cos(angle), torch.sin(angle)
            x, y = frame[:, 0].clone(), frame[:, 1].clone()
            frame[:, 0] = cosine * x - sine * y
            frame[:, 1] = sine * x + cosine * y
        if training and jitter_std > 0:
            frame = frame + torch.randn_like(frame) * float(jitter_std)
        moved.append(frame)
    return moved


__all__ = [
    "KITTIRouterDataset",
    "calibrate_lambdas",
    "collate_raw",
    "move_and_augment",
    "training_scales",
]
