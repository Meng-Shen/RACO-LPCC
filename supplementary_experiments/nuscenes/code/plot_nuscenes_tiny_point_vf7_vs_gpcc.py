#!/usr/bin/env python3
"""Plot the selected nuScenes TinyPoint-VF7 router against fixed G-PCC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_auc(rows: list[dict[str, str]], x_min: float, x_max: float) -> float:
    x = np.asarray([float(row["measured_bpp"]) for row in rows])
    y = np.asarray([float(row["mAP"]) for row in rows])
    order = np.argsort(x)
    grid = np.linspace(x_min, x_max, 1024)
    curve = np.interp(grid, x[order], y[order])
    return float(np.trapz(curve, grid) / (x_max - x_min))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-curve-csv", required=True)
    parser.add_argument("--tiny-selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    selection_dir = Path(args.tiny_selection_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(
        (selection_dir / "MAP_SELECTION_COMPLETE.json").read_text()
    )
    checkpoint = selection["best"]["checkpoint"]
    tiny = sorted(
        [
            row
            for row in read_rows(selection_dir / "candidate_curves.csv")
            if row["checkpoint"] == checkpoint
        ],
        key=lambda row: float(row["measured_bpp"]),
    )
    fixed = sorted(
        [
            row
            for row in read_rows(args.fixed_curve_csv)
            if row["series"] == "fixed_quantization"
        ],
        key=lambda row: float(row["measured_bpp"]),
    )
    if len(fixed) != 6 or len(tiny) != 6:
        raise RuntimeError(
            f"Expected six points per curve, got fixed={len(fixed)}, tiny={len(tiny)}"
        )

    combined: list[dict[str, str]] = []
    for series, rows in (("fixed_gpcc", fixed), ("tiny_point_vf7", tiny)):
        for row in rows:
            combined.append(
                {
                    "series": series,
                    "checkpoint": checkpoint if series == "tiny_point_vf7" else "",
                    "measured_bpp": row["measured_bpp"],
                    "mAP": row["mAP"],
                    "NDS": row.get("NDS", ""),
                    "lambda": row.get("lambda", ""),
                    "qstep_mm": row.get("qstep_mm", ""),
                    "selection_counts": row.get("selection_counts", ""),
                }
            )
    csv_path = output_dir / "nuscenes_tiny_point_vf7_vs_gpcc_map_bpp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    fig, ax = plt.subplots(figsize=(7.7, 5.3), dpi=190)
    for rows, label, marker, color in (
        (fixed, "Fixed G-PCC", "o", "#4C78A8"),
        (tiny, "TinyPoint-VF7 route", "o", "#E45756"),
    ):
        ax.plot(
            [float(row["measured_bpp"]) for row in rows],
            [100.0 * float(row["mAP"]) for row in rows],
            marker=marker,
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )
    ax.set_xlabel("BPP (total G-PCC bits / total original points)")
    ax.set_ylabel("nuScenes mAP (%)")
    ax.set_title("nuScenes CenterPoint: TinyPoint-VF7 vs Fixed G-PCC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = output_dir / "nuscenes_tiny_point_vf7_vs_gpcc_map_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    x_min = min(float(row["measured_bpp"]) for row in fixed)
    x_max = max(float(row["measured_bpp"]) for row in fixed)
    fixed_auc = normalized_auc(fixed, x_min, x_max)
    tiny_auc = normalized_auc(tiny, x_min, x_max)
    summary = {
        "metric": "official nuScenes 10-class mAP",
        "x_axis": "linear BPP = total G-PCC bits / total original points",
        "tiny_point_vf7_checkpoint": checkpoint,
        "tiny_point_vf7_selection": selection["best"],
        "common_range_normalized_map_bpp_auc": {
            "fixed_gpcc": fixed_auc,
            "tiny_point_vf7": tiny_auc,
        },
        "tiny_vs_fixed_auc_gain": tiny_auc - fixed_auc,
        "tiny_vs_fixed_auc_relative_percent": 100.0 * (tiny_auc / fixed_auc - 1.0),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
