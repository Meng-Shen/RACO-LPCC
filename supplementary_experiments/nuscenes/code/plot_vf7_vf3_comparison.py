#!/usr/bin/env python3
"""Plot fixed G-PCC, TinyPoint-VF7, and TinyPoint-VF3 on one BPP curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KITTI_COLUMNS = (
    "Car_3d_AP_R40_moderate",
    "Pedestrian_3d_AP_R40_moderate",
    "Cyclist_3d_AP_R40_moderate",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collapse_curve(points: list[tuple[float, float]]) -> np.ndarray:
    grouped: dict[float, list[float]] = {}
    for x, y in points:
        grouped.setdefault(float(x), []).append(float(y))
    return np.asarray(
        sorted((x, float(np.mean(values))) for x, values in grouped.items()),
        dtype=np.float64,
    )


def load_curve(task: str, source: Path, role: str) -> tuple[np.ndarray, dict]:
    metadata: dict = {"source": str(source.resolve())}
    if task == "nuscenes" and source.is_dir():
        selection = json.loads((source / "MAP_SELECTION_COMPLETE.json").read_text())
        checkpoint = selection["best"]["checkpoint"]
        rows = [
            row for row in read_rows(source / "candidate_curves.csv")
            if row["checkpoint"] == checkpoint
        ]
        metadata.update({"checkpoint": checkpoint, "selection": selection["best"]})
    else:
        rows = read_rows(source)
        if task == "nuscenes" and "series" in rows[0]:
            if role == "baseline":
                rows = [row for row in rows if "fixed" in row["series"].lower()]
            else:
                expected = f"tiny_point_{role}"
                selected = [row for row in rows if row["series"].lower() == expected]
                if not selected:
                    selected = [row for row in rows if "fixed" not in row["series"].lower()]
                rows = selected

    if task == "kitti":
        points = [
            (float(row["bpp"]), float(np.mean([float(row[name]) for name in KITTI_COLUMNS])))
            for row in rows
        ]
    elif task == "semantickitti":
        points = [
            (float(row["bpp"]), float(row.get("decoded_miou", row["miou"])) * 100.0)
            for row in rows
        ]
    elif task == "nuscenes":
        points = [(float(row["measured_bpp"]), float(row["mAP"]) * 100.0) for row in rows]
    else:
        raise ValueError(task)
    curve = collapse_curve(points)
    if curve.shape != (6, 2):
        raise RuntimeError(f"Expected exactly six {role} points, got shape={curve.shape} from {source}")
    return curve, metadata


def normalized_log_bpp_auc(curve: np.ndarray, lo: float, hi: float) -> float:
    x = np.log(np.maximum(curve[:, 0], 1e-12))
    grid = np.linspace(lo, hi, 2001)
    return float(np.trapz(np.interp(grid, x, curve[:, 1]), grid) / (hi - lo))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["kitti", "semantickitti", "nuscenes"], required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--vf7", type=Path, required=True)
    parser.add_argument("--vf3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    sources = {}
    for role, source in (("baseline", args.baseline), ("vf7", args.vf7), ("vf3", args.vf3)):
        curves[role], sources[role] = load_curve(args.task, source, role)

    lo = max(float(np.log(curve[:, 0]).min()) for curve in curves.values())
    hi = min(float(np.log(curve[:, 0]).max()) for curve in curves.values())
    if not hi > lo:
        raise RuntimeError(f"No common BPP interval: lo={lo}, hi={hi}")
    auc = {key: normalized_log_bpp_auc(curve, lo, hi) for key, curve in curves.items()}

    task_labels = {
        "kitti": ("KITTI PV-RCNN", "Moderate 3-class mAP_R40 (%)", "kitti"),
        "semantickitti": ("SemanticKITTI sequence 08 MinkUNet", "mIoU (%)", "semantickitti"),
        "nuscenes": ("nuScenes CenterPoint", "mAP (%)", "nuscenes"),
    }
    title, ylabel, stem = task_labels[args.task]
    styles = {
        "baseline": ("Fixed whole-frame G-PCC", "s", "#4C78A8"),
        "vf7": ("TinyPoint-VF7", "o", "#E45756"),
        "vf3": ("TinyPoint-VF3", "^", "#54A24B"),
    }
    fig, ax = plt.subplots(figsize=(7.9, 5.5), dpi=210)
    for role in ("baseline", "vf7", "vf3"):
        label, marker, color = styles[role]
        curve = curves[role]
        ax.plot(curve[:, 0], curve[:, 1], marker=marker, linewidth=2.2,
                markersize=6.2, color=color, label=label)
    ax.set_xlabel("BPP (total G-PCC bits / total original points)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: VF7 vs VF3 routing")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend()
    fig.tight_layout()
    png_path = args.output_dir / f"{stem}_gpcc_vf7_vf3_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output_dir / f"{stem}_gpcc_vf7_vf3_bpp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "series", "rate_id", "bpp", "metric_percent"])
        writer.writeheader()
        for role in ("baseline", "vf7", "vf3"):
            for index, (bpp, metric) in enumerate(curves[role]):
                writer.writerow({"task": args.task, "series": role, "rate_id": index,
                                 "bpp": f"{bpp:.12g}", "metric_percent": f"{metric:.12g}"})

    summary = {
        "status": "complete",
        "task": args.task,
        "six_points_per_curve": True,
        "x_axis": "total G-PCC bits / total original points",
        "common_log_bpp_interval": [float(np.exp(lo)), float(np.exp(hi))],
        "normalized_metric_auc_percent": auc,
        "vf7_minus_baseline_auc": auc["vf7"] - auc["baseline"],
        "vf3_minus_baseline_auc": auc["vf3"] - auc["baseline"],
        "vf7_minus_vf3_auc": auc["vf7"] - auc["vf3"],
        "vf7_improves_over_vf3": auc["vf7"] > auc["vf3"],
        "sources": sources,
        "csv": str(csv_path.resolve()),
        "plot": str(png_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
