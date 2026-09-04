#!/usr/bin/env python3
"""Plot ShapeNet55 fixed G-PCC, full-router reference, and TinyPoint router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "fixed_gpcc": "Fixed G-PCC",
    "full_sparse_router": "Full sparse router (reference)",
    "tiny_point_router": "TinyPoint router (new)",
}


def mean_class_accuracy(labels, predictions):
    recalls = []
    for class_id in range(55):
        mask = labels == class_id
        if mask.any():
            recalls.append(float((predictions[mask] == labels[mask]).mean()))
    return float(np.mean(recalls))


def adaptive_rows(name, labels, class_predictions, test_indices, router):
    if not np.array_equal(test_indices, router["indices"].astype(np.int64)):
        raise RuntimeError(f"Test sample order mismatch for {name}")
    selected_all = router["selected_levels"].astype(np.int64)
    true_bpp = router["true_bpp"].astype(np.float64)
    lambdas = router["lambdas"].astype(np.float64)
    sample_rows = np.arange(len(labels))
    rows = []
    for rate_id, lam in enumerate(lambdas):
        selected = selected_all[:, rate_id]
        predictions = class_predictions[sample_rows, selected]
        rows.append({
            "series": name,
            "lambda": float(lam),
            "qstep": "",
            "measured_gpcc_bpp": float(true_bpp[sample_rows, selected].mean()),
            "overall_accuracy": float((predictions == labels).mean()),
            "mean_class_accuracy": mean_class_accuracy(labels, predictions),
            "selection_counts": ";".join(
                str(int(value)) for value in np.bincount(selected, minlength=6)
            ),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--test-indices", type=Path, required=True)
    parser.add_argument("--full-router", type=Path, required=True)
    parser.add_argument("--tiny-router", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    quant = np.load(args.quant)
    test_indices = np.load(args.test_indices).astype(np.int64)
    row_by_index = {int(index): row for row, index in enumerate(quant["indices"])}
    test_rows = np.asarray([row_by_index[int(index)] for index in test_indices])
    labels = quant["labels"][test_rows].astype(np.int64)
    predictions = quant["predictions"][test_rows].astype(np.int64)
    qsteps = quant["qsteps"].astype(np.float64)
    full = np.load(args.full_router)
    tiny = np.load(args.tiny_router)
    fixed_bpp = tiny["true_bpp"].astype(np.float64)

    rows = []
    for level, qstep in enumerate(qsteps):
        current = predictions[:, level]
        rows.append({
            "series": "fixed_gpcc",
            "lambda": "",
            "qstep": float(qstep),
            "measured_gpcc_bpp": float(fixed_bpp[:, level].mean()),
            "overall_accuracy": float((current == labels).mean()),
            "mean_class_accuracy": mean_class_accuracy(labels, current),
            "selection_counts": "",
        })
    rows.extend(adaptive_rows("full_sparse_router", labels, predictions, test_indices, full))
    rows.extend(adaptive_rows("tiny_point_router", labels, predictions, test_indices, tiny))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "shapenet55_tinypoint_accuracy_bpp_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(7.6, 5.4), dpi=220)
    styles = {
        "fixed_gpcc": ("o", "#4C78A8", "--"),
        "full_sparse_router": ("s", "#59A14F", "-"),
        "tiny_point_router": ("^", "#E45756", "-"),
    }
    for series in ("fixed_gpcc", "full_sparse_router", "tiny_point_router"):
        current = sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: float(row["measured_gpcc_bpp"]),
        )
        marker, color, linestyle = styles[series]
        axis.plot(
            [float(row["measured_gpcc_bpp"]) for row in current],
            [100.0 * float(row["overall_accuracy"]) for row in current],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            markersize=6,
            label=LABELS[series],
        )
    axis.set_xlabel("Measured G-PCC bitrate (bits/original point)")
    axis.set_ylabel("Overall accuracy (%)")
    axis.set_title("ShapeNet55 / Point-MAE test: Accuracy–BPP")
    axis.grid(True, alpha=0.28)
    axis.legend(loc="lower right")
    fig.tight_layout()
    png_path = output / "shapenet55_tinypoint_accuracy_bpp_comparison.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": "ShapeNet55 official test split",
        "samples": int(len(labels)),
        "classifier": "Point-MAE geometry-only",
        "series": list(LABELS),
        "bitrate": "total measured G-PCC bits / total original points",
        "csv": str(csv_path),
        "png": str(png_path),
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
