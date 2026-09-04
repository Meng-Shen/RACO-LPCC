#!/usr/bin/env python3
"""Calibrate lambdas and route SUN RGB-D with TinyPoint on 160 mm cell means."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tiny_point_absolute_loss_monotonic_rate_proxy import (
    TinyPointAbsoluteLossMonotonicRateProxy,
)
from train_sunrgbd_tiny_point_router_ddp import (
    QSTEPS_MM,
    SUNRGBDTinyPointDataset,
    collate_variable,
    dense_batch,
    read_ids,
)


class SUNRGBDTinyPointInferenceDataset(Dataset):
    def __init__(self, cache_dir: Path, split_file: Path, bpp_csv: Path):
        self.scene_ids = read_ids(split_file)
        cached_ids = np.load(cache_dir / "val_scene_ids.npy", allow_pickle=False).tolist()
        if cached_ids != self.scene_ids:
            raise RuntimeError("160 mm cache scene order does not match official val split")
        self.points = np.load(cache_dir / "val_points_160mm.npy", mmap_mode="r")
        self.offsets = np.load(cache_dir / "val_offsets_160mm.npy", mmap_mode="r")
        rates = {}
        with bpp_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rates.setdefault(row["scene_id"], np.full(6, np.nan, np.float32))[
                    int(row["rate_id"])
                ] = float(row["bpp"])
        missing = [sid for sid in self.scene_ids
                   if sid not in rates or not np.isfinite(rates[sid]).all()]
        if missing:
            raise RuntimeError(f"Missing validation BPP for {len(missing)} scenes")
        self.rates = rates

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        scene_id = self.scene_ids[index]
        return {
            "scene_id": scene_id,
            "points": torch.from_numpy(np.asarray(self.points[start:end]).copy()),
            "bpp": torch.from_numpy(self.rates[scene_id].copy()),
        }


def collate_inference(batch):
    return {
        "scene_ids": [item["scene_id"] for item in batch],
        "points": [item["points"] for item in batch],
        "bpp": torch.stack([item["bpp"] for item in batch]),
    }


@torch.no_grad()
def predict(model, dataset, batch_size, workers, device, point_cloud_range, collate_fn):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn, persistent_workers=workers > 0,
    )
    scene_ids, predicted_loss, predicted_bpp, true_loss, true_bpp = [], [], [], [], []
    point_counts = []
    model.eval()
    for batch_index, batch in enumerate(loader):
        points, valid_mask, lengths = dense_batch(
            batch["points"], device, False, point_cloud_range
        )
        output = model(points, valid_mask)
        scene_ids.extend(batch["scene_ids"])
        point_counts.extend(lengths)
        predicted_loss.append(output["loss_pred"].cpu().numpy())
        predicted_bpp.append(output["bpp_pred"].cpu().numpy())
        if "loss" in batch:
            true_loss.append(batch["loss"].numpy())
        if "raw_bpp" in batch:
            true_bpp.append(batch["raw_bpp"].numpy())
        else:
            true_bpp.append(batch["bpp"].numpy())
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(json.dumps({
                "batch": batch_index + 1,
                "scenes_seen": len(scene_ids),
                "input_points_mean": float(np.mean(lengths)),
                "input_points_max": int(max(lengths)),
            }), flush=True)
    return {
        "scene_ids": scene_ids,
        "predicted_loss": np.concatenate(predicted_loss),
        "predicted_bpp": np.concatenate(predicted_bpp),
        "true_loss": np.concatenate(true_loss) if true_loss else None,
        "true_bpp": np.concatenate(true_bpp),
        "point_counts": np.asarray(point_counts),
    }


def selected_levels(loss: np.ndarray, bpp: np.ndarray, lambdas: np.ndarray):
    return (loss[:, None, :] + lambdas[None, :, None] * bpp[:, None, :]).argmin(axis=2)


def calibrate_lambdas(train_predictions):
    loss = train_predictions["predicted_loss"].astype(np.float64)
    bpp = train_predictions["predicted_bpp"].astype(np.float64)
    base = max(np.median(np.ptp(loss, axis=1)) /
               max(np.median(np.ptp(bpp, axis=1)), 1e-6), 1e-6)
    candidates = np.concatenate([[0.0], np.logspace(
        np.log10(base) - 5.0, np.log10(base) + 5.0, 2000
    )])
    levels = selected_levels(loss, bpp, candidates)
    true_bpp = train_predictions["true_bpp"].astype(np.float64)
    achieved = np.take_along_axis(
        true_bpp[:, None, :], levels[:, :, None], axis=2
    ).squeeze(2).mean(axis=0)
    targets = np.geomspace(max(float(achieved.min()), 1e-12),
                           max(float(achieved.max()), 1e-12), 6)
    chosen = []
    signatures = set()
    for target in targets:
        distances = np.abs(np.log(np.maximum(achieved, 1e-12)) - np.log(target))
        for candidate_index in np.argsort(distances):
            index = int(candidate_index)
            signature = tuple(np.bincount(levels[:, index], minlength=6).tolist())
            if index not in chosen and signature not in signatures:
                chosen.append(index)
                signatures.add(signature)
                break
    if len(chosen) != 6:
        raise RuntimeError(f"Could only calibrate {len(chosen)} distinct routing points")
    chosen.sort(key=lambda index: achieved[index])
    lambdas = candidates[chosen]
    selected = levels[:, chosen]
    return lambdas, {
        "lambda_scale_base": base,
        "lambdas_low_rate_to_high_rate": lambdas.tolist(),
        "train_aggregate_true_bpp": achieved[chosen].tolist(),
        "train_mean_selected_level": selected.mean(axis=0).tolist(),
        "train_selection_counts": [
            np.bincount(selected[:, i], minlength=6).tolist() for i in range(6)
        ],
        "calibration_uses_test": False,
        "calibration_split": "full official SUN RGB-D train (5285 scenes)",
    }


def write_predictions(path: Path, predictions, lambdas):
    levels = selected_levels(predictions["predicted_loss"], predictions["predicted_bpp"], lambdas)
    fields = ["scene_id"]
    fields += [f"pred_loss_L{i}" for i in range(6)]
    fields += [f"pred_bpp_L{i}" for i in range(6)]
    fields += [f"selected_level_R{i}" for i in range(6)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, scene_id in enumerate(predictions["scene_ids"]):
            row = {"scene_id": scene_id}
            row.update({f"pred_loss_L{i}": predictions["predicted_loss"][row_index, i]
                        for i in range(6)})
            row.update({f"pred_bpp_L{i}": predictions["predicted_bpp"][row_index, i]
                        for i in range(6)})
            row.update({f"selected_level_R{i}": int(levels[row_index, i])
                        for i in range(6)})
            writer.writerow(row)
    report = {
        "scenes": len(predictions["scene_ids"]),
        "bpp_monotonic_violation_rate": float(
            (np.diff(predictions["predicted_bpp"], axis=1) < 0).mean()
        ),
        "selection_counts": [
            np.bincount(levels[:, i], minlength=6).tolist() for i in range(6)
        ],
        "input_point_count": {
            "mean": float(predictions["point_counts"].mean()),
            "median": float(np.median(predictions["point_counts"])),
            "p95": float(np.percentile(predictions["point_counts"], 95)),
            "min": int(predictions["point_counts"].min()),
            "max": int(predictions["point_counts"].max()),
        },
        "output": str(path.resolve()),
    }
    if predictions["true_loss"] is not None:
        report["loss_mae"] = float(np.abs(
            predictions["predicted_loss"] - predictions["true_loss"]
        ).mean())
    report["bpp_mae"] = float(np.abs(
        predictions["predicted_bpp"] - predictions["true_bpp"]
    ).mean())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--val-split", required=True, type=Path)
    parser.add_argument("--train-loss-csv", required=True, type=Path)
    parser.add_argument("--train-bpp-csv", required=True, type=Path)
    parser.add_argument("--val-bpp-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    train_set = SUNRGBDTinyPointDataset(
        args.cache_dir, args.train_split, args.train_loss_csv, args.train_bpp_csv
    )
    val_set = SUNRGBDTinyPointInferenceDataset(
        args.cache_dir, args.val_split, args.val_bpp_csv
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyPointAbsoluteLossMonotonicRateProxy(
        256, checkpoint["loss_scales"], checkpoint["mean_log_bpp"], input_channels=3
    )
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device)
    point_cloud_range = checkpoint["args"]["point_cloud_range"]

    train_predictions = predict(
        model, train_set, args.batch_size, args.workers, device,
        point_cloud_range, collate_variable
    )
    lambdas, calibration = calibrate_lambdas(train_predictions)
    val_predictions = predict(
        model, val_set, args.batch_size, args.workers, device,
        point_cloud_range, collate_inference
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_report = write_predictions(
        args.output_dir / "train_router_predictions.csv", train_predictions, lambdas
    )
    val_report = write_predictions(
        args.output_dir / "val_router_predictions.csv", val_predictions, lambdas
    )
    payload = {
        "status": "complete",
        "qsteps_mm_coarse_to_fine": QSTEPS_MM,
        "router_input": "160 mm detector-aligned cells represented by original-point mean XYZ",
        "routing_rule": "argmin predicted_loss + lambda * predicted_bpp",
        "calibration": calibration,
        "train": train_report,
        "val": val_report,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    (args.output_dir / "lambda_calibration_and_metrics.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
