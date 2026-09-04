#!/usr/bin/env python3
"""Plot ShapeNet55 Point-MAE fixed-quantization and learned-routing curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean_class_accuracy(labels, predictions):
    recalls = []
    for class_id in range(55):
        mask = labels == class_id
        if mask.any():
            recalls.append(float((predictions[mask] == labels[mask]).mean()))
    return float(np.mean(recalls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-quant", required=True)
    parser.add_argument("--router-predictions", required=True)
    parser.add_argument("--test-indices", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    quant = np.load(args.test_quant)
    router = np.load(args.router_predictions)
    test_indices = np.load(args.test_indices).astype(np.int64)
    row_by_index = {int(index): row for row, index in enumerate(quant["indices"])}
    rows_for_test = np.asarray([row_by_index[int(index)] for index in test_indices])
    if not np.array_equal(test_indices, router["indices"]):
        raise RuntimeError("Test sample order mismatch")
    labels = quant["labels"][rows_for_test].astype(np.int64)
    class_predictions = quant["predictions"][rows_for_test].astype(np.int64)
    qsteps = quant["qsteps"].astype(float)
    true_bpp = router["true_bpp"].astype(float)
    rows = []
    for level, qstep in enumerate(qsteps):
        pred = class_predictions[:, level]
        rows.append({
            "series": "fixed_quantization",
            "lambda": "",
            "qstep": qstep,
            "measured_gpcc_bpp": float(true_bpp[:, level].mean()),
            "overall_accuracy": float((pred == labels).mean()),
            "mean_class_accuracy": mean_class_accuracy(labels, pred),
            "selection_counts": "",
        })

    selected_all = router["selected_levels"].astype(np.int64)
    lambdas = router["lambdas"].astype(float)
    sample_rows = np.arange(len(labels))
    for position, lam in enumerate(lambdas):
        selected = selected_all[:, position]
        pred = class_predictions[sample_rows, selected]
        rows.append({
            "series": "learned_rate_aware_router",
            "lambda": lam,
            "qstep": "",
            "measured_gpcc_bpp": float(true_bpp[sample_rows, selected].mean()),
            "overall_accuracy": float((pred == labels).mean()),
            "mean_class_accuracy": mean_class_accuracy(labels, pred),
            "selection_counts": ";".join(str(int(value)) for value in np.bincount(selected, minlength=6)),
        })

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pointmae_shapenet55_accuracy_bpp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    for axis, metric, title in [
        (axes[0], "overall_accuracy", "ShapeNet55 test: overall accuracy"),
        (axes[1], "mean_class_accuracy", "ShapeNet55 test: mean class accuracy"),
    ]:
        for series, label, marker, color in [
            ("fixed_quantization", "Fixed quantization", "o", "#4C78A8"),
            ("learned_rate_aware_router", "Learned routing proxy", "^", "#E45756"),
        ]:
            current = sorted(
                [row for row in rows if row["series"] == series],
                key=lambda row: float(row["measured_gpcc_bpp"]),
            )
            axis.plot(
                [float(row["measured_gpcc_bpp"]) for row in current],
                [100.0 * float(row[metric]) for row in current],
                marker=marker, linewidth=2, markersize=5.5, color=color, label=label,
            )
        axis.set_xlabel("Measured G-PCC bitrate (bits/original point)")
        axis.set_ylabel("Accuracy (%)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Point-MAE geometry-only: fixed quantization vs learned routing")
    fig.tight_layout()
    png_path = output / "pointmae_shapenet55_accuracy_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": "ShapeNet55 official test split",
        "samples": int(len(labels)),
        "classifier": "Point-MAE, XYZ only, official ShapeNet55 self-supervised initialization then 55-class fine-tuning",
        "qsteps_coarse_to_fine": qsteps.tolist(),
        "bitrate": "sum per-object G-PCC bits / sum original points",
        "oracle_curve_plotted": False,
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
