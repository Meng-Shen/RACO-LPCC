#!/usr/bin/env python3
"""Evaluate exact nuScenes mAP--BPP for fixed and rate-aware quantization."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmengine
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, METRICS


QSTEPS_MM = (2048, 1024, 512, 256, 128, 64)

_WORKER_CFG = None
_WORKER_DATASET_META = None
_WORKER_RECORDS = None
_WORKER_JSON_ROOT = None


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
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


def load_records(root: str | Path) -> list[dict]:
    paths = sorted(Path(root).resolve().glob("shard_*/predictions.pkl"))
    if not paths:
        raise FileNotFoundError(f"No prediction shards below {root}")
    records: list[dict] = []
    for path in paths:
        current = mmengine.load(path)
        print(f"Loaded {len(current)} records from {path}", flush=True)
        records.extend(current)
    records.sort(key=lambda row: int(row["dataset_index"]))
    indices = [int(row["dataset_index"]) for row in records]
    if indices != list(range(len(records))):
        raise RuntimeError(
            f"Prediction cache is incomplete or duplicated: "
            f"records={len(records)}, first={indices[:3]}, last={indices[-3:]}"
        )
    if any(len(row["predictions"]) != 6 for row in records):
        raise RuntimeError("Every sample must contain six rate predictions")
    return records


def metric_value(metrics: dict, suffix: str) -> float:
    matches = [
        float(value)
        for key, value in metrics.items()
        if str(key).endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one /{suffix}, got {list(metrics)}")
    return matches[0]


def evaluate_selection(cfg, dataset_meta, records, selected, output_prefix):
    evaluator_cfg = dict(cfg.test_evaluator)
    evaluator_cfg["jsonfile_prefix"] = str(output_prefix)
    evaluator = METRICS.build(evaluator_cfg)
    evaluator.dataset_meta = dataset_meta
    for record, level in zip(records, selected):
        cached = record["predictions"][int(level)]
        # The exporter stores MMEngine InstanceData objects.  NuScenesMetric in
        # this mmdet3d build expects ordinary dictionaries and iterates their
        # keys; passing InstanceData directly makes it iterate samples instead.
        pred_3d = cached["pred_instances_3d"]
        pred_2d = cached["pred_instances"]
        if hasattr(pred_3d, "to_dict"):
            pred_3d = pred_3d.to_dict()
        else:
            pred_3d = dict(pred_3d)
        if hasattr(pred_2d, "to_dict"):
            pred_2d = pred_2d.to_dict()
        else:
            pred_2d = dict(pred_2d)
        evaluator.process({}, [{
            "pred_instances_3d": pred_3d,
            "pred_instances": pred_2d,
            "sample_idx": cached["sample_idx"],
        }])
    metrics = evaluator.evaluate(len(records))
    return metrics, metric_value(metrics, "mAP"), metric_value(metrics, "NDS")


def exact_bpp(proxy_rows: list[dict], selected: np.ndarray):
    points = np.asarray([int(row["num_points"]) for row in proxy_rows])
    bpp = np.asarray(
        [[float(row[f"L{level}_true_bpp"]) for level in range(6)]
         for row in proxy_rows],
        dtype=np.float64,
    )
    frame = np.arange(len(proxy_rows))
    bits = float(np.sum(bpp[frame, selected] * points))
    total_points = int(points.sum())
    return bits / total_points, bits, total_points


def evaluate_job(job: dict) -> dict:
    """Evaluate one fixed/routed operating point in a forked CPU process."""
    _, map_value, nds = evaluate_selection(
        _WORKER_CFG,
        _WORKER_DATASET_META,
        _WORKER_RECORDS,
        job["selected"],
        _WORKER_JSON_ROOT / job["name"],
    )
    return {"job_index": job["job_index"], "mAP": map_value, "NDS": nds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--rate-aware-predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=25)
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args()

    cfg = Config.fromfile(str(Path(args.config).resolve()))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet3d")
    records = load_records(args.prediction_root)
    proxy_by_index = {
        int(float(row["scene_id"])): row
        for row in read_rows(args.rate_aware_predictions_csv)
    }
    if sorted(proxy_by_index) != list(range(len(records))):
        raise RuntimeError(
            f"Proxy rows do not cover prediction indices: {len(proxy_by_index)}"
        )
    proxy_rows = [proxy_by_index[index] for index in range(len(records))]

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    dataset.full_init()
    if len(dataset) != len(records):
        raise RuntimeError(
            f"Dataset/cache length mismatch: {len(dataset)} != {len(records)}"
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_root = output / "nuscenes_result_json"
    json_root.mkdir(exist_ok=True)
    rows: list[dict] = []
    jobs: list[dict] = []

    for level, qstep in enumerate(QSTEPS_MM):
        selected = np.full(len(records), level, dtype=np.int64)
        bpp, bits, points = exact_bpp(proxy_rows, selected)
        rows.append({
            "series": "fixed_quantization",
            "lambda": "",
            "qstep_mm": qstep,
            "measured_bpp": bpp,
            "total_bits": bits,
            "total_original_points": points,
            "mAP": "",
            "NDS": "",
            "selection_counts": ";".join(
                str(len(records) if i == level else 0) for i in range(6)
            ),
        })
        jobs.append({
            "job_index": len(rows) - 1,
            "name": f"fixed_L{level}",
            "selected": selected,
        })

    for lam_index in range(6):
        lam = float(proxy_rows[0][f"lambda_{lam_index}"])
        selected = np.asarray([
            int(row[f"lambda_{lam_index}_predicted_level"])
            for row in proxy_rows
        ])
        bpp, bits, points = exact_bpp(proxy_rows, selected)
        counts = np.bincount(selected, minlength=6)
        rows.append({
            "series": "rate_aware_proxy",
            "lambda": lam,
            "qstep_mm": "",
            "measured_bpp": bpp,
            "total_bits": bits,
            "total_original_points": points,
            "mAP": "",
            "NDS": "",
            "selection_counts": ";".join(str(int(value)) for value in counts),
        })
        jobs.append({
            "job_index": len(rows) - 1,
            "name": f"rate_aware_lambda_{lam_index}",
            "selected": selected,
        })

    global _WORKER_CFG, _WORKER_DATASET_META, _WORKER_RECORDS, _WORKER_JSON_ROOT
    _WORKER_CFG = cfg
    _WORKER_DATASET_META = dataset.metainfo
    _WORKER_RECORDS = records
    _WORKER_JSON_ROOT = json_root
    worker_count = max(1, min(int(args.parallel_workers), len(jobs)))
    print(f"Evaluating {len(jobs)} operating points with {worker_count} workers", flush=True)
    if worker_count == 1:
        results = [evaluate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            futures = [executor.submit(evaluate_job, job) for job in jobs]
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"Completed point {len(results)}/{len(jobs)}: "
                    f"job={result['job_index']} mAP={result['mAP']:.6f} "
                    f"NDS={result['NDS']:.6f}",
                    flush=True,
                )
    for result in results:
        rows[int(result["job_index"])]["mAP"] = float(result["mAP"])
        rows[int(result["job_index"])]["NDS"] = float(result["NDS"])

    csv_path = output / "nuscenes_rate_aware_measured_gpcc_map_bpp.csv"
    write_rows(csv_path, rows)
    fig, ax = plt.subplots(figsize=(7.4, 5.1), dpi=180)
    for series, label, marker, color in (
        ("fixed_quantization", "Fixed quantization", "o", "#4C78A8"),
        ("rate_aware_proxy", f"Rate-aware routing (epoch {args.checkpoint_epoch})", "^", "#E45756"),
    ):
        current = sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: float(row["measured_bpp"]),
        )
        ax.plot(
            [float(row["measured_bpp"]) for row in current],
            [100.0 * float(row["mAP"]) for row in current],
            marker=marker,
            linewidth=2,
            markersize=6,
            color=color,
            label=label,
        )
    ax.set_xlabel("Measured G-PCC bitrate (bits per original point)")
    ax.set_ylabel("nuScenes mAP (%)")
    ax.set_title("nuScenes XYZ-only CenterPoint: mAP–BPP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = output / "nuscenes_rate_aware_measured_gpcc_map_bpp.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "validation_frames": len(records),
        "detector": "CenterPoint XYZ-only single keyframe",
        "routing_checkpoint_epoch": args.checkpoint_epoch,
        "qsteps_mm_coarse_to_fine": list(QSTEPS_MM),
        "bitrate_method": "sum(frame bits) / sum(original frame points)",
        "metric": "official nuScenes 10-class mAP",
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
