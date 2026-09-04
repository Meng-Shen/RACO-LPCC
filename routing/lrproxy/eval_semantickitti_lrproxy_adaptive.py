#!/usr/bin/env python3
"""Evaluate only LRproxy-selected SemanticKITTI rates on sequence 08.

The fixed six-rate G-PCC curve is reused from the complete-router evaluation.
For each frame this evaluator runs MinkUNet only at the unique rates selected
by the six lambdas, avoiding an unnecessary full six-rate recomputation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


RACO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACO_ROOT / "integrations" / "mmdetection3d" / "tools"))

from eval_semantickitti_decoded_miou_router_rd import (  # noqa: E402
    DEFAULT_STEPS_MM,
    NUM_CLASSES,
    build_label_lookup,
    build_model,
    confusion_matrix,
    load_gpcc_table,
    load_points_and_labels,
    load_router_labels,
    load_val_items,
    metrics_from_confusion,
    nearest_neighbor_transfer,
    predict_quantized_labels,
    quantize_xyz,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--router-manifest", required=True, type=Path)
    parser.add_argument("--gpcc-details", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-world-size", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def log(message):
    rank = int(os.environ.get("RANK", "0"))
    print(f"[{time.strftime('%F %T')}][rank {rank}] {message}", flush=True)


def evaluate_worker(args, items):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world != args.expected_world_size:
        raise RuntimeError(f"Expected {args.expected_world_size} workers, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    expected_ids = [item["frame_id"] for item in items]
    _, thresholds, selections = load_router_labels(
        args.router_manifest, expected_ids
    )
    worker_items = items[rank::world]
    label_lookup = build_label_lookup(args.config)
    model = build_model(args.config, args.checkpoint, device)
    adaptive_source = np.zeros((6, NUM_CLASSES, NUM_CLASSES), np.int64)
    adaptive_decoded = np.zeros_like(adaptive_source)
    selection_counts = np.zeros((6, 6), np.int64)
    completed_ids = []
    inference_count = 0
    started = time.time()

    for index, item in enumerate(worker_items, 1):
        original_xyz, raw_labels = load_points_and_labels(item)
        ground_truth = label_lookup[(raw_labels & 0xFFFF).astype(np.int64)]
        labels = [selections[rate_id][item["frame_id"]] for rate_id in range(6)]
        frame_results = {}
        for label in sorted(set(labels)):
            quantized_xyz = quantize_xyz(original_xyz, DEFAULT_STEPS_MM[label])
            quantized_prediction = predict_quantized_labels(
                model, quantized_xyz, device
            )
            original_prediction = nearest_neighbor_transfer(
                original_xyz, quantized_xyz, quantized_prediction
            )
            source_matrix = confusion_matrix(original_prediction, ground_truth)
            decoded_ground_truth = nearest_neighbor_transfer(
                quantized_xyz, original_xyz, ground_truth
            )
            decoded_matrix = confusion_matrix(
                quantized_prediction, decoded_ground_truth
            )
            frame_results[label] = (source_matrix, decoded_matrix)
            inference_count += 1
        for rate_id, label in enumerate(labels):
            source_matrix, decoded_matrix = frame_results[label]
            adaptive_source[rate_id] += source_matrix
            adaptive_decoded[rate_id] += decoded_matrix
            selection_counts[rate_id, label] += 1
        completed_ids.append(item["frame_id"])
        if index == 1 or index % 25 == 0:
            elapsed = time.time() - started
            fps = index / max(elapsed, 1e-9)
            eta = (len(worker_items) - index) / max(fps, 1e-9)
            log(
                f"{index}/{len(worker_items)} frames, {inference_count} selected-rate "
                f"inferences; ETA {eta / 60.0:.1f} min"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = args.output_dir / f"adaptive_shard_{rank:02d}_of_{world:02d}.npz"
    np.savez_compressed(
        shard,
        adaptive_source_confusion=adaptive_source,
        adaptive_decoded_confusion=adaptive_decoded,
        selection_counts=selection_counts,
        frame_ids=np.asarray(completed_ids, dtype="U6"),
        thresholds=np.asarray(thresholds, np.float64),
        steps_mm=np.asarray(DEFAULT_STEPS_MM, np.int64),
        selected_rate_inferences=np.asarray([inference_count], np.int64),
    )
    log(f"Saved {shard}")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_shards(args, items):
    expected_ids = [item["frame_id"] for item in items]
    manifest, thresholds, selections = load_router_labels(
        args.router_manifest, expected_ids
    )
    adaptive_source = np.zeros((6, NUM_CLASSES, NUM_CLASSES), np.int64)
    adaptive_decoded = np.zeros_like(adaptive_source)
    selection_counts = np.zeros((6, 6), np.int64)
    completed = []
    total_inferences = 0
    shards = []
    for rank in range(args.expected_world_size):
        path = args.output_dir / (
            f"adaptive_shard_{rank:02d}_of_{args.expected_world_size:02d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(path)
        with np.load(path) as payload:
            if tuple(payload["steps_mm"].tolist()) != tuple(DEFAULT_STEPS_MM):
                raise RuntimeError(f"Step mismatch in {path}")
            if not np.allclose(payload["thresholds"], thresholds):
                raise RuntimeError(f"Lambda mismatch in {path}")
            adaptive_source += payload["adaptive_source_confusion"]
            adaptive_decoded += payload["adaptive_decoded_confusion"]
            selection_counts += payload["selection_counts"]
            completed.extend(payload["frame_ids"].tolist())
            total_inferences += int(payload["selected_rate_inferences"][0])
    if len(completed) != len(set(completed)) or set(completed) != set(expected_ids):
        raise RuntimeError(
            f"Shard coverage mismatch: unique={len(set(completed))}, "
            f"expected={len(expected_ids)}"
        )
    if args.gpcc_details is None or not args.gpcc_details.is_file():
        raise FileNotFoundError(args.gpcc_details)
    gpcc = load_gpcc_table(args.gpcc_details, expected_ids)
    total_points = sum(
        gpcc[(frame_id, 0)]["num_points"] for frame_id in expected_ids
    )
    rows = []
    for rate_id, threshold in enumerate(thresholds):
        source_stats = metrics_from_confusion(adaptive_source[rate_id])
        decoded_stats = metrics_from_confusion(adaptive_decoded[rate_id])
        total_bits = sum(
            gpcc[(frame_id, selections[rate_id][frame_id])]["bits"]
            for frame_id in expected_ids
        )
        row = {
            "rate_id": rate_id,
            "threshold": threshold,
            "bpp": total_bits / total_points,
            "miou": decoded_stats["miou"],
            "accuracy": decoded_stats["accuracy"],
            "decoded_miou": decoded_stats["miou"],
            "decoded_accuracy": decoded_stats["accuracy"],
            "source_miou": source_stats["miou"],
            "source_accuracy": source_stats["accuracy"],
            "num_frames": len(expected_ids),
            "total_points": total_points,
            "total_bits": total_bits,
        }
        for label in range(6):
            row[f"label_{label}_frames"] = int(selection_counts[rate_id, label])
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lrproxy_guided_miou_bpp.csv"
    write_csv(csv_path, rows)
    summary = {
        "status": "complete",
        "completed_at": time.strftime("%F %T"),
        "evaluation_split": "SemanticKITTI sequence 08",
        "num_frames": len(expected_ids),
        "primary_metric": "global decoded-domain 19-class mIoU",
        "bpp": "total G-PCC bits / total original points",
        "router_manifest": str(args.router_manifest.resolve()),
        "router_checkpoint": manifest.get("ckpt", ""),
        "segmentation_checkpoint": str(args.checkpoint.resolve()),
        "adaptive_csv": str(csv_path.resolve()),
        "selected_rate_inferences": total_inferences,
        "full_six_rate_inferences_avoided": 6 * len(expected_ids) - total_inferences,
        "shards": [str(path.resolve()) for path in shards],
    }
    (args.output_dir / "LRPROXY_ADAPTIVE_EVAL_COMPLETE.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.router_manifest = args.router_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    items = load_val_items(args.dataset_root, args.max_frames)
    if args.merge_only:
        merge_shards(args, items)
    else:
        evaluate_worker(args, items)


if __name__ == "__main__":
    main()
