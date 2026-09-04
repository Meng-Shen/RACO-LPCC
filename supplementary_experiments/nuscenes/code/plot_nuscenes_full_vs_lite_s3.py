#!/usr/bin/env python3
"""Plot fixed G-PCC, a reference full router, and Lite-S3 on linear BPP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def selected_rows(selection_dir):
    selection_dir = Path(selection_dir)
    summary = json.loads((selection_dir / "MAP_SELECTION_COMPLETE.json").read_text())
    stem = summary["best"]["checkpoint"]
    rows = [
        row for row in read_rows(selection_dir / "candidate_curves.csv")
        if row["checkpoint"] == stem
    ]
    if len(rows) != 6:
        raise RuntimeError(f"Expected six selected rows for {stem}, got {len(rows)}")
    return stem, summary, sorted(rows, key=lambda row: float(row["measured_bpp"]))


def normalized_auc(rows, x_min, x_max):
    x = np.asarray([float(row["measured_bpp"]) for row in rows])
    y = np.asarray([float(row["mAP"]) for row in rows])
    order = np.argsort(x)
    grid = np.linspace(x_min, x_max, 1024)
    curve = np.interp(grid, x[order], y[order])
    return float(np.trapz(curve, grid) / (x_max - x_min))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-curve-csv", required=True)
    parser.add_argument("--full-selection-dir", required=True)
    parser.add_argument("--lite-selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixed = sorted(
        [row for row in read_rows(args.fixed_curve_csv) if row["series"] == "fixed_quantization"],
        key=lambda row: float(row["measured_bpp"]),
    )
    full_stem, full_summary, full = selected_rows(args.full_selection_dir)
    lite_stem, lite_summary, lite = selected_rows(args.lite_selection_dir)
    x_min = min(float(row["measured_bpp"]) for row in fixed)
    x_max = max(float(row["measured_bpp"]) for row in fixed)
    aucs = {
        "fixed_gpcc": normalized_auc(fixed, x_min, x_max),
        "reference_full_router": normalized_auc(full, x_min, x_max),
        "lite_s3_sixloss_monotonic_router": normalized_auc(lite, x_min, x_max),
    }

    combined = []
    for series, checkpoint, rows in (
        ("fixed_gpcc", "", fixed),
        ("reference_full_router", full_stem, full),
        ("lite_s3_sixloss_monotonic_router", lite_stem, lite),
    ):
        for row in rows:
            combined.append({
                "series": series,
                "checkpoint": checkpoint,
                "measured_bpp": row["measured_bpp"],
                "mAP": row["mAP"],
                "NDS": row.get("NDS", ""),
                "lambda": row.get("lambda", ""),
                "qstep_mm": row.get("qstep_mm", ""),
                "selection_counts": row.get("selection_counts", ""),
            })
    csv_path = output / "nuscenes_full_vs_lite_s3_map_bpp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    fig, ax = plt.subplots(figsize=(7.7, 5.3), dpi=190)
    for rows, label, marker, color in (
        (fixed, "Fixed G-PCC", "o", "#4C78A8"),
        (full, "Reference full router", "s", "#E45756"),
        (lite, "Lite-S3 monotonic router", "^", "#54A24B"),
    ):
        ax.plot(
            [float(row["measured_bpp"]) for row in rows],
            [100.0 * float(row["mAP"]) for row in rows],
            marker=marker,
            linewidth=2.1,
            markersize=6,
            color=color,
            label=label,
        )
    ax.set_xlabel("BPP (total G-PCC bits / total original points)")
    ax.set_ylabel("nuScenes mAP (%)")
    ax.set_title("nuScenes CenterPoint: Reference Full vs Lite-S3 Router")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = output / "nuscenes_full_vs_lite_s3_map_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "metric": "official nuScenes 10-class mAP",
        "x_axis": "linear BPP = total G-PCC bits / total original points",
        "full_checkpoint": full_stem,
        "lite_checkpoint": lite_stem,
        "full_selection": full_summary["best"],
        "lite_selection": lite_summary["best"],
        "common_range_normalized_map_bpp_auc": aucs,
        "lite_vs_full_auc_relative_percent": 100.0 * (aucs["lite_s3_sixloss_monotonic_router"] / aucs["reference_full_router"] - 1.0),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
