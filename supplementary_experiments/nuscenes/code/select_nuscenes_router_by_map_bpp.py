#!/usr/bin/env python3
"""Select a router checkpoint by official validation mAP--BPP curve area."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS

import evaluate_nuscenes_multisweep_rate_aware_map_bpp as evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--candidate-predictions-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--fixed-curve-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--single-checkpoint-evaluation", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_checkpoint(training_dir: Path, stem: str) -> Path:
    if stem == "candidate_init":
        return training_dir / "candidate_init.pth"
    return training_dir / "checkpoints" / f"{stem}.pth"


def curve_auc_gain(fixed_x, fixed_y, routed_x, routed_y):
    order = np.argsort(routed_x)
    routed_x = np.asarray(routed_x)[order]
    routed_y = np.asarray(routed_y)[order]
    unique_x = np.unique(routed_x)
    unique_y = np.asarray([
        routed_y[routed_x == value].max() for value in unique_x
    ])
    grid = np.linspace(float(fixed_x.min()), float(fixed_x.max()), 512)
    fixed_curve = np.interp(grid, fixed_x, fixed_y)
    routed_curve = np.interp(grid, unique_x, unique_y)
    return float(np.trapz(routed_curve - fixed_curve, grid) / (grid[-1] - grid[0]))


def main():
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    candidates_dir = Path(args.candidate_predictions_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_root = output_dir / "candidate_result_json"
    json_root.mkdir(exist_ok=True)

    fixed_rows = [
        row for row in read_rows(args.fixed_curve_csv)
        if row["series"] == "fixed_quantization"
    ]
    fixed_rows.sort(key=lambda row: float(row["measured_bpp"]))
    fixed_x = np.asarray([float(row["measured_bpp"]) for row in fixed_rows])
    fixed_y = np.asarray([float(row["mAP"]) for row in fixed_rows])

    cfg = Config.fromfile(str(Path(args.config).resolve()))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet3d")
    records = evaluation.load_records(args.prediction_root)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    dataset.full_init()
    if len(dataset) != len(records):
        raise RuntimeError(f"Dataset/cache mismatch: {len(dataset)} != {len(records)}")

    csv_paths = sorted(candidates_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No candidate prediction CSVs in {candidates_dir}")
    if args.single_checkpoint_evaluation and len(csv_paths) != 1:
        raise RuntimeError(
            "--single-checkpoint-evaluation requires exactly one training-selected checkpoint, "
            f"found {len(csv_paths)}"
        )
    jobs = []
    candidate_info = {}
    for csv_path in csv_paths:
        by_index = {
            int(float(row["scene_id"])): row for row in read_rows(csv_path)
        }
        if sorted(by_index) != list(range(len(records))):
            raise RuntimeError(f"Incomplete candidate predictions: {csv_path}")
        proxy_rows = [by_index[index] for index in range(len(records))]
        points = []
        for lam_index in range(6):
            selected = np.asarray([
                int(row[f"lambda_{lam_index}_predicted_level"])
                for row in proxy_rows
            ], dtype=np.int64)
            bpp, bits, total_points = evaluation.exact_bpp(proxy_rows, selected)
            job_index = len(jobs)
            jobs.append({
                "job_index": job_index,
                "name": f"{csv_path.stem}/lambda_{lam_index}",
                "selected": selected,
            })
            points.append({
                "job_index": job_index,
                "lambda_index": lam_index,
                "lambda": float(proxy_rows[0][f"lambda_{lam_index}"]),
                "measured_bpp": bpp,
                "total_bits": bits,
                "total_original_points": total_points,
                "selection_counts": ";".join(
                    str(int(value))
                    for value in np.bincount(selected, minlength=6)
                ),
            })
        candidate_info[csv_path.stem] = {
            "prediction_csv": csv_path,
            "proxy_rows": proxy_rows,
            "points": points,
        }

    evaluation._WORKER_CFG = cfg
    evaluation._WORKER_DATASET_META = dataset.metainfo
    evaluation._WORKER_RECORDS = records
    evaluation._WORKER_JSON_ROOT = json_root
    worker_count = max(1, min(int(args.parallel_workers), len(jobs)))
    print(
        f"Evaluating {len(csv_paths)} checkpoints x 6 lambdas = "
        f"{len(jobs)} jobs with {worker_count} workers",
        flush=True,
    )
    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        futures = [executor.submit(evaluation.evaluate_job, job) for job in jobs]
        results = {}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[int(result["job_index"])] = result
            completed += 1
            print(
                f"completed={completed}/{len(jobs)} "
                f"job={result['job_index']} mAP={result['mAP']:.6f}",
                flush=True,
            )

    curve_rows = []
    score_rows = []
    for stem, info in candidate_info.items():
        checkpoint_path = candidate_checkpoint(training_dir, stem)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        routed_x, routed_y = [], []
        for point in info["points"]:
            result = results[point["job_index"]]
            routed_x.append(point["measured_bpp"])
            routed_y.append(float(result["mAP"]))
            curve_rows.append({
                "checkpoint": stem,
                **{key: value for key, value in point.items() if key != "job_index"},
                "mAP": float(result["mAP"]),
                "NDS": float(result["NDS"]),
            })
        auc_gain = curve_auc_gain(
            fixed_x, fixed_y, np.asarray(routed_x), np.asarray(routed_y)
        )
        metrics = checkpoint.get("metrics", {})
        score_rows.append({
            "checkpoint": stem,
            "epoch": int(checkpoint.get("epoch", 0)),
            "map_bpp_auc_gain": auc_gain,
            "map_bpp_auc_gain_percentage_points": 100.0 * auc_gain,
            "normalized_val_total_loss": metrics.get("total_loss", ""),
            "val_loss_mae": metrics.get("loss_mae", ""),
            "val_bpp_mae": metrics.get("bpp_mae", ""),
        })

    score_rows.sort(key=lambda row: float(row["map_bpp_auc_gain"]), reverse=True)
    best = score_rows[0]
    best_stem = best["checkpoint"]
    best_checkpoint = candidate_checkpoint(training_dir, best_stem)
    best_predictions = candidate_info[best_stem]["prediction_csv"]
    shutil.copy2(best_checkpoint, training_dir / "best_map_bpp.pth")
    shutil.copy2(
        best_predictions, training_dir / "test_rate_aware_predictions_map_selected.csv"
    )
    write_rows(output_dir / "candidate_scores.csv", score_rows)
    write_rows(output_dir / "candidate_curves.csv", curve_rows)
    summary = {
        "selection_metric": (
            "minimum full-training regression total loss; official validation is final evaluation only"
            if args.single_checkpoint_evaluation
            else "mean official nuScenes mAP gain over the fixed-quantization piecewise-linear curve, integrated across the measured BPP range"
        ),
        "official_validation_used_for_checkpoint_selection": not args.single_checkpoint_evaluation,
        "candidate_count": len(score_rows),
        "best": best,
        "best_checkpoint": str(training_dir / "best_map_bpp.pth"),
        "best_predictions": str(
            training_dir / "test_rate_aware_predictions_map_selected.csv"
        ),
    }
    (output_dir / "MAP_SELECTION_COMPLETE.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
