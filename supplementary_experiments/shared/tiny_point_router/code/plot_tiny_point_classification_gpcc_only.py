#!/usr/bin/env python3
"""Plot fixed G-PCC and the latest TinyPoint classification router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def class_accuracy(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    recalls = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            recalls.append(float((predictions[mask] == labels[mask]).mean()))
    return float(np.mean(recalls))


def normalized_log_auc(rows) -> float:
    rows = sorted(rows, key=lambda row: float(row["bpp"]))
    x = np.log(np.maximum([float(row["bpp"]) for row in rows], 1e-12))
    y = np.asarray([float(row["accuracy"]) for row in rows])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", required=True, type=Path)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    quant = np.load(args.quant)
    router = np.load(args.router)
    test_indices = router["indices"].astype(np.int64)
    quant_indices = quant["indices"].astype(np.int64)
    row_by_index = {int(index): row for row, index in enumerate(quant_indices)}
    quant_rows = np.asarray([row_by_index[int(index)] for index in test_indices])
    labels = quant["labels"][quant_rows].astype(np.int64)
    predictions = quant["predictions"][quant_rows].astype(np.int64)
    qsteps = quant["qsteps"].astype(np.float64)
    true_bpp = router["true_bpp"].astype(np.float64)
    selected_levels = router["selected_levels"].astype(np.int64)
    lambdas = router["lambdas"].astype(np.float64)
    sample_rows = np.arange(len(test_indices))

    rows = []
    for level, qstep in enumerate(qsteps):
        current = predictions[:, level]
        rows.append({
            "series": "fixed_gpcc",
            "rate_point": level,
            "lambda": "",
            "qstep": float(qstep),
            "bpp": float(true_bpp[:, level].mean()),
            "accuracy": float((current == labels).mean()),
            "mean_class_accuracy": class_accuracy(labels, current, args.num_classes),
            "selection_counts": "",
        })
    for rate_point, lam in enumerate(lambdas):
        selected = selected_levels[:, rate_point]
        current = predictions[sample_rows, selected]
        rows.append({
            "series": "tiny_point_router",
            "rate_point": rate_point,
            "lambda": float(lam),
            "qstep": "",
            "bpp": float(true_bpp[sample_rows, selected].mean()),
            "accuracy": float((current == labels).mean()),
            "mean_class_accuracy": class_accuracy(labels, current, args.num_classes),
            "selection_counts": ";".join(
                str(int(value)) for value in np.bincount(selected, minlength=6)
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.prefix}_tinypoint_vs_gpcc_accuracy_bpp.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.8, 5.45), dpi=220)
    styles = {
        "fixed_gpcc": ("Fixed G-PCC", "o", "--", "#4c78a8"),
        "tiny_point_router": ("TinyPoint router (all-train)", "^", "-", "#e4572e"),
    }
    for series in ("fixed_gpcc", "tiny_point_router"):
        current = sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: float(row["bpp"]),
        )
        label, marker, linestyle, color = styles[series]
        ax.plot(
            [float(row["bpp"]) for row in current],
            [100.0 * float(row["accuracy"]) for row in current],
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.3,
            markersize=6.5,
            label=label,
        )
    ax.set_xlabel("BPP (total G-PCC bits / total original points)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(f"{args.dataset} / {args.classifier}: TinyPoint vs fixed G-PCC")
    ax.grid(True, linestyle=":", alpha=0.42)
    ax.legend(loc="lower right")
    fig.tight_layout()
    png_path = args.output_dir / f"{args.prefix}_tinypoint_vs_gpcc_accuracy_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    grouped = {
        series: [row for row in rows if row["series"] == series]
        for series in ("fixed_gpcc", "tiny_point_router")
    }
    summary = {
        "dataset": args.dataset,
        "classifier": args.classifier,
        "samples": int(len(test_indices)),
        "bpp": "total measured G-PCC bits / total original points",
        "normalized_log_bpp_accuracy_auc": {
            series: normalized_log_auc(current) for series, current in grouped.items()
        },
        "png": str(png_path),
        "csv": str(csv_path),
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
