#!/usr/bin/env python3
"""Evaluate a saved nuScenes rate-aware proxy without stopping training.

This evaluates prediction fidelity on the official validation split and draws
the immediately available rate-distortion curve from the measured per-frame
G-PCC rates and CenterPoint task-loss labels.  It deliberately does not call
this curve mAP: exact mAP needs the six-rate detector prediction cache.
"""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_scannet_rate_aware_proxy import (
    RateAwareSparseProxy,
    export_predictions,
    flexible_load,
    make_loader,
    run_epoch,
)


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rd(prediction_rows: list[dict]) -> list[dict]:
    points = np.asarray([int(row["num_points"]) for row in prediction_rows])
    total_points = int(points.sum())
    true_loss = np.asarray(
        [[float(row[f"L{i}_true_delta"]) for i in range(6)]
         for row in prediction_rows],
        dtype=np.float64,
    )
    true_bpp = np.asarray(
        [[float(row[f"L{i}_true_bpp"]) for i in range(6)]
         for row in prediction_rows],
        dtype=np.float64,
    )
    rows: list[dict] = []
    for level, qstep in enumerate(QSTEPS_MM):
        rows.append({
            "series": "fixed_quantization",
            "lambda": "",
            "qstep_mm": qstep,
            "measured_bpp": float(np.sum(true_bpp[:, level] * points) / total_points),
            "mean_centerpoint_loss_delta": float(np.mean(true_loss[:, level])),
            "selection_accuracy": "",
            "selection_counts": ";".join(
                str(len(prediction_rows) if i == level else 0) for i in range(6)
            ),
        })

    for lam_index in range(6):
        lam = float(prediction_rows[0][f"lambda_{lam_index}"])
        predicted = np.asarray([
            int(row[f"lambda_{lam_index}_predicted_level"])
            for row in prediction_rows
        ])
        oracle = np.asarray([
            int(row[f"lambda_{lam_index}_oracle_level"])
            for row in prediction_rows
        ])
        frame_index = np.arange(len(prediction_rows))
        rows.append({
            "series": "rate_aware_proxy",
            "lambda": lam,
            "qstep_mm": "",
            "measured_bpp": float(
                np.sum(true_bpp[frame_index, predicted] * points) / total_points
            ),
            "mean_centerpoint_loss_delta": float(
                np.mean(true_loss[frame_index, predicted])
            ),
            "selection_accuracy": float(np.mean(predicted == oracle)),
            "selection_counts": ";".join(
                str(int(value))
                for value in np.bincount(predicted, minlength=6)
            ),
        })
    return rows


def draw_plot(rows: list[dict], path: Path, epoch: int) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.1), dpi=180)
    for series, label, marker, color in (
        ("fixed_quantization", "Fixed quantization", "o", "#4C78A8"),
        ("rate_aware_proxy", f"Rate-aware proxy (best epoch {epoch})", "^", "#E45756"),
    ):
        current = sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: float(row["measured_bpp"]),
        )
        ax.plot(
            [float(row["measured_bpp"]) for row in current],
            [float(row["mean_centerpoint_loss_delta"]) for row in current],
            marker=marker,
            linewidth=2,
            markersize=6,
            color=color,
            label=label,
        )
    ax.set_xlabel("Measured G-PCC bitrate (bits per original point)")
    ax.set_ylabel("Mean CenterPoint loss increase vs 64 mm (lower is better)")
    ax.set_title("nuScenes official validation: proxy rate-distortion test")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--points-dir", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--test-loss-csv", required=True)
    parser.add_argument("--test-bpp-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = Namespace(**checkpoint["args"])
    train_args.points_dir = args.points_dir
    train_args.batch_size = args.batch_size
    train_args.workers = args.workers
    train_args.jitter_std = 0.0

    loader, dataset = make_loader(
        train_args,
        args.test_split,
        args.test_loss_csv,
        args.test_bpp_csv,
        False,
    )
    device = torch.device("cuda:0")
    model = RateAwareSparseProxy(
        dataset.spatial_shape,
        int(train_args.feat_dim),
        dataset.mean_log_bpp,
    ).to(device)
    flexible_load(model, checkpoint["model"])
    lambdas = torch.tensor(checkpoint["lambdas"], dtype=torch.float32, device=device)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_csv = output / "test_rate_aware_predictions.csv"
    metrics = run_epoch(model, loader, device, lambdas, train_args)
    export_predictions(model, loader, device, lambdas, train_args, prediction_csv)
    prediction_rows = read_rows(prediction_csv)
    rd_rows = summarize_rd(prediction_rows)
    rd_csv = output / "nuscenes_current_best_loss_bpp.csv"
    plot_path = output / "nuscenes_current_best_loss_bpp.png"
    write_rows(rd_csv, rd_rows)
    draw_plot(rd_rows, plot_path, int(checkpoint["epoch"]))
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "official_validation_samples": len(prediction_rows),
        "test_metrics": metrics,
        "routing_rule": checkpoint.get("routing_rule"),
        "bitrate_method": "sum(frame_bpp * original_points) / sum(original_points)",
        "curve_metric": "mean CenterPoint task-loss delta relative to 64 mm",
        "note": "Exact mAP requires six-rate CenterPoint prediction caching.",
        "predictions_csv": str(prediction_csv),
        "rd_csv": str(rd_csv),
        "plot": str(plot_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
