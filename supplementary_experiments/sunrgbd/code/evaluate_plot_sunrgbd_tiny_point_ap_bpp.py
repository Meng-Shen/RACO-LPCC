#!/usr/bin/env python3
"""Official SUN RGB-D AP-BPP: fixed G-PCC versus TinyPoint 160 mm cell-mean router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmengine
import numpy as np
from mmengine.logging import MMLogger
from mmdet3d.datasets import SUNRGBDDataset
from mmdet3d.evaluation import indoor_eval
from mmdet3d.structures import get_box_type


def evaluate(records, levels, logger):
    annotations = [record["eval_ann_info"] for record in records]
    predictions = [record["predictions"][int(level)]
                   for record, level in zip(records, levels)]
    _, box_mode = get_box_type("depth")
    return indoor_eval(
        annotations, predictions, [0.25, 0.5], SUNRGBDDataset.METAINFO["classes"],
        logger=logger, box_mode_3d=box_mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--router-csv", required=True, type=Path)
    parser.add_argument("--gpcc-csv", required=True, type=Path)
    parser.add_argument("--lambda-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-scenes", type=int, default=5050)
    parser.add_argument("--expected-shards", type=int, default=7)
    args = parser.parse_args()

    paths = sorted(args.prediction_root.glob("shard_*/predictions.pkl"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} shards, got {len(paths)}")
    records, manifests = [], []
    for path in paths:
        records.extend(mmengine.load(path))
        manifests.append(json.loads(path.with_suffix(".manifest.json").read_text()))
    records.sort(key=lambda row: int(row["dataset_index"]))
    if len(records) != args.expected_scenes:
        raise RuntimeError(f"Expected {args.expected_scenes} records, got {len(records)}")
    if [int(row["dataset_index"]) for row in records] != list(range(args.expected_scenes)):
        raise RuntimeError("Prediction indices do not exactly cover SUN RGB-D val")
    scene_ids = [row["scene_id"] for row in records]

    with args.router_csv.open(newline="") as handle:
        router = {row["scene_id"]: row for row in csv.DictReader(handle)}
    if set(scene_ids) - router.keys():
        raise RuntimeError("Router predictions miss validation scenes")
    gpcc = {}
    with args.gpcc_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            gpcc[(row["scene_id"], int(row["rate_id"]))] = (
                int(row["bits"]), int(row["num_points"])
            )
    if any((sid, level) not in gpcc for sid in scene_ids for level in range(6)):
        raise RuntimeError("G-PCC CSV does not contain all val scene/rate pairs")

    calibration = json.loads(args.lambda_json.read_text())["calibration"]
    lambdas = calibration["lambdas_low_rate_to_high_rate"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = MMLogger.get_instance(
        "sunrgbd_tiny_point_ap_bpp",
        log_file=str(args.output_dir / "evaluation.log"), log_level="INFO",
    )
    rows = []
    for level, qstep in enumerate(manifests[0]["qsteps_mm_coarse_to_fine"]):
        levels = np.full(args.expected_scenes, level, dtype=np.int64)
        metrics = evaluate(records, levels, logger)
        total_bits = sum(gpcc[(sid, level)][0] for sid in scene_ids)
        total_points = sum(gpcc[(sid, level)][1] for sid in scene_ids)
        rows.append({
            "series": "fixed_gpcc_baseline",
            "rate_point": level,
            "qstep_mm": qstep,
            "lambda": "",
            "bpp": total_bits / total_points,
            "mAP_0.25": metrics["mAP_0.25"],
            "mAP_0.50": metrics["mAP_0.50"],
            "selection_counts": ";".join(
                str(args.expected_scenes if i == level else 0) for i in range(6)
            ),
        })
    for rate_point in range(6):
        levels = np.asarray([
            int(router[sid][f"selected_level_R{rate_point}"]) for sid in scene_ids
        ], dtype=np.int64)
        metrics = evaluate(records, levels, logger)
        total_bits = sum(gpcc[(sid, int(level))][0]
                         for sid, level in zip(scene_ids, levels))
        total_points = sum(gpcc[(sid, int(level))][1]
                           for sid, level in zip(scene_ids, levels))
        rows.append({
            "series": "tiny_point_160mm_router",
            "rate_point": rate_point,
            "qstep_mm": "",
            "lambda": lambdas[rate_point],
            "bpp": total_bits / total_points,
            "mAP_0.25": metrics["mAP_0.25"],
            "mAP_0.50": metrics["mAP_0.50"],
            "selection_counts": ";".join(
                str(int(value)) for value in np.bincount(levels, minlength=6)
            ),
        })

    csv_path = args.output_dir / "sunrgbd_tiny_point_160mm_vs_gpcc_ap_bpp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    for axis, key, title in (
        (axes[0], "mAP_0.25", "SUN RGB-D mAP@0.25"),
        (axes[1], "mAP_0.50", "SUN RGB-D mAP@0.50"),
    ):
        for series, label, color, marker in (
            ("fixed_gpcc_baseline", "Fixed G-PCC", "#4C78A8", "o"),
            ("tiny_point_160mm_router", "TinyPoint router (160 mm cell means)", "#E45756", "s"),
        ):
            selected = sorted([row for row in rows if row["series"] == series],
                              key=lambda row: float(row["bpp"]))
            axis.plot(
                [float(row["bpp"]) for row in selected],
                [100.0 * float(row[key]) for row in selected],
                color=color, marker=marker, linewidth=2.2, markersize=5.5, label=label,
            )
        axis.set_xlabel("BPP (total G-PCC bits / total original points)")
        axis.set_ylabel("mAP (%)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    png_path = args.output_dir / "sunrgbd_tiny_point_160mm_vs_gpcc_ap_bpp.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "status": "complete",
        "dataset": "SUN RGB-D official val",
        "curves_shown": ["fixed G-PCC baseline", "TinyPoint router (160 mm cell means)"],
        "bpp_definition": "sum selected encoded geometry bits / sum original points",
        "test_used_for_lambda_or_epoch_selection": False,
        "csv": str(csv_path.resolve()),
        "png": str(png_path.resolve()),
    }
    (args.output_dir / "EVALUATION_COMPLETE.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": manifest, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
