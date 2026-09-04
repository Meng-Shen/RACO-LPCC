#!/usr/bin/env python3
"""Evaluate a decoded-domain segmentation router on SemanticKITTI sequence 08.

The proxy selects one of six uniform G-PCC geometry steps for every frame and
loss threshold.  The primary mIoU is computed directly on decoded points after
nearest-neighbour ground-truth recolouring.  Original-point prediction transfer
is accumulated as an audit metric. Independent torchrun workers accumulate
confusion matrices; merge mode combines them with measured G-PCC bit counts.
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

from _bootstrap import MMDET_ROOT, bootstrap_paths

bootstrap_paths()

import numpy as np
import torch


DEFAULT_STEPS_MM = (2048, 1024, 512, 256, 128, 64)
NUM_CLASSES = 19
IGNORE_LABEL = 19


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--router-manifest", type=Path)
    parser.add_argument("--gpcc-details", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps-mm", nargs=6, type=int, default=DEFAULT_STEPS_MM)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def log(message):
    rank = int(os.environ.get("RANK", "0"))
    print(f"[{time.strftime('%F %T')}][rank {rank}] {message}", flush=True)


def normalize_frame_id(value):
    value = str(value).strip()
    if "_" in value:
        value = value.rsplit("_", 1)[-1]
    return value.zfill(6)


def load_val_items(dataset_root, max_frames=0):
    info_path = dataset_root / "semantickitti_infos_val.pkl"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    with info_path.open("rb") as handle:
        payload = pickle.load(handle)
    items = []
    for raw in payload["data_list"]:
        sample_idx = str(raw["sample_idx"])
        if not sample_idx.startswith("08_"):
            continue
        point_path = dataset_root / raw["lidar_points"]["lidar_path"]
        label_path = dataset_root / raw["pts_semantic_mask_path"]
        frame_id = normalize_frame_id(sample_idx)
        if not point_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing validation data for {sample_idx}")
        if point_path.stem != frame_id:
            raise RuntimeError(f"Frame/path mismatch: {sample_idx}, {point_path}")
        items.append(
            {
                "sample_idx": sample_idx,
                "frame_id": frame_id,
                "point_path": point_path,
                "label_path": label_path,
            }
        )
    items.sort(key=lambda item: item["frame_id"])
    if max_frames > 0:
        items = items[:max_frames]
    if not items:
        raise RuntimeError("No labeled SemanticKITTI sequence-08 frames found")
    frame_ids = [item["frame_id"] for item in items]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("Duplicate validation frame IDs")
    return items


def prepare_proxy_split(args, items):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "semantickitti_sequence08.txt"
    with split_path.open("w") as handle:
        for item in items:
            handle.write(item["frame_id"] + "\n")
    metadata_path = args.output_dir / "sequence08_split.json"
    metadata_path.write_text(
        json.dumps(
            {
                "split": "SemanticKITTI sequence 08 (labeled validation/test split)",
                "num_frames": len(items),
                "velodyne_dir": str(items[0]["point_path"].parent.resolve()),
                "split_file": str(split_path.resolve()),
            },
            indent=2,
        )
    )
    log(f"Prepared {len(items)} validation frame IDs: {split_path}")


def load_router_labels(manifest_path, expected_ids):
    manifest = json.loads(manifest_path.read_text())
    label_csvs = manifest.get("label_csvs", [])
    if len(label_csvs) != 6:
        raise RuntimeError(f"Expected six router label CSVs, got {len(label_csvs)}")
    thresholds = [float(item["threshold"]) for item in label_csvs]
    if thresholds != sorted(thresholds):
        raise ValueError("Detection-loss thresholds are not sorted")
    selections = []
    expected = set(expected_ids)
    for rate_id, item in enumerate(label_csvs):
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        mapping = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                frame_id = normalize_frame_id(row["frame_id"])
                label = int(row["jucp_label"])
                if label < 0 or label >= 6:
                    raise ValueError(f"Invalid label {label} for {frame_id}")
                if frame_id in mapping:
                    raise RuntimeError(f"Duplicate router label for {frame_id}")
                mapping[frame_id] = label
        missing = sorted(expected - set(mapping))
        extras = sorted(set(mapping) - expected)
        if missing or extras:
            raise RuntimeError(
                f"Router rate {rate_id} coverage mismatch: missing={len(missing)}, "
                f"extras={len(extras)}, first_missing={missing[:3]}"
            )
        selections.append(mapping)
    return manifest, thresholds, selections


def add_repo_to_path(config_path):
    repo_root = MMDET_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def build_label_lookup(config_path):
    add_repo_to_path(config_path)
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    mapping = dict(cfg.metainfo["seg_label_mapping"])
    lookup = np.full(1 << 16, IGNORE_LABEL, dtype=np.int64)
    for source, target in mapping.items():
        lookup[int(source)] = int(target)
    return lookup


def build_model(config_path, checkpoint_path, device):
    add_repo_to_path(config_path)
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    if int(cfg.model.backbone.in_channels) != 3:
        raise ValueError("Expected the XYZ-only three-channel MinkUNet")
    if int(cfg.model.decode_head.num_classes) != NUM_CLASSES:
        raise ValueError("Expected a 19-class segmentation head")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model.to(device)
    model.eval()
    return model


def load_points_and_labels(item):
    points = np.fromfile(item["point_path"], dtype=np.float32)
    if points.size % 4:
        raise ValueError(f"Invalid point cloud: {item['point_path']}")
    points = points.reshape(-1, 4)
    raw_labels = np.fromfile(item["label_path"], dtype=np.uint32)
    if len(points) != len(raw_labels):
        raise ValueError(f"Point/label mismatch for {item['sample_idx']}")
    return points[:, :3].astype(np.float32, copy=False), raw_labels


def quantize_xyz(xyz_m, step_mm):
    if len(xyz_m) == 0:
        return np.empty((0, 3), dtype=np.float32)
    xyz_mm = np.rint(xyz_m.astype(np.float64) * 1000.0).astype(np.int64)
    offset_mm = xyz_mm.min(axis=0, keepdims=True)
    lattice = np.rint((xyz_mm - offset_mm) / float(step_mm)).astype(np.int64)
    unique_lattice = np.unique(lattice, axis=0)
    decoded_mm = unique_lattice * int(step_mm) + offset_mm
    return (decoded_mm.astype(np.float64) / 1000.0).astype(np.float32)


@torch.no_grad()
def predict_quantized_labels(model, quantized_xyz, device):
    from mmdet3d.structures import Det3DDataSample, PointData

    points = torch.from_numpy(quantized_xyz).to(device=device, dtype=torch.float32)
    sample = Det3DDataSample()
    sample.gt_pts_seg = PointData(
        pts_semantic_mask=torch.full(
            (len(points),), IGNORE_LABEL, dtype=torch.long, device=device
        )
    )
    output = model.test_step(
        {"inputs": {"points": [points]}, "data_samples": [sample]}
    )[0]
    prediction = output.pred_pts_seg.pts_semantic_mask
    prediction = prediction.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(prediction) != len(quantized_xyz):
        raise RuntimeError("Prediction/quantized-point length mismatch")
    return prediction


def nearest_neighbor_transfer(original_xyz, quantized_xyz, prediction):
    from scipy.spatial import cKDTree

    if len(quantized_xyz) == 0:
        raise RuntimeError("Empty quantized point cloud")
    tree = cKDTree(quantized_xyz.astype(np.float64, copy=False))
    _, nearest = tree.query(
        original_xyz.astype(np.float64, copy=False), k=1, workers=1
    )
    return prediction[np.asarray(nearest, dtype=np.int64)]


def confusion_matrix(prediction, ground_truth):
    valid = ground_truth != IGNORE_LABEL
    prediction = prediction[valid]
    ground_truth = ground_truth[valid]
    if len(ground_truth) == 0:
        raise RuntimeError("Frame has no valid semantic points")
    if prediction.min() < 0 or prediction.max() >= NUM_CLASSES:
        raise ValueError("Prediction is outside the 19 valid classes")
    return np.bincount(
        ground_truth * NUM_CLASSES + prediction,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def evaluate_worker(args, items):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != args.expected_world_size:
        raise RuntimeError(f"Expected {args.expected_world_size} workers, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    expected_ids = [item["frame_id"] for item in items]
    _, thresholds, selections = load_router_labels(args.router_manifest, expected_ids)
    worker_items = items[rank::world_size]
    label_lookup = build_label_lookup(args.config)
    model = build_model(args.config, args.checkpoint, device)

    fixed_source = np.zeros((6, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    adaptive_source = np.zeros_like(fixed_source)
    fixed_decoded = np.zeros_like(fixed_source)
    adaptive_decoded = np.zeros_like(fixed_source)
    selection_counts = np.zeros((6, 6), dtype=np.int64)
    completed_ids = []
    start = time.time()
    for index, item in enumerate(worker_items, 1):
        original_xyz, raw_labels = load_points_and_labels(item)
        ground_truth = label_lookup[(raw_labels & 0xFFFF).astype(np.int64)]
        frame_source_confusions = []
        frame_decoded_confusions = []
        for label, step_mm in enumerate(args.steps_mm):
            quantized_xyz = quantize_xyz(original_xyz, step_mm)
            quantized_prediction = predict_quantized_labels(
                model, quantized_xyz, device
            )
            original_prediction = nearest_neighbor_transfer(
                original_xyz, quantized_xyz, quantized_prediction
            )
            source_matrix = confusion_matrix(
                original_prediction, ground_truth)
            decoded_ground_truth = nearest_neighbor_transfer(
                quantized_xyz, original_xyz, ground_truth)
            decoded_matrix = confusion_matrix(
                quantized_prediction, decoded_ground_truth)
            fixed_source[label] += source_matrix
            fixed_decoded[label] += decoded_matrix
            frame_source_confusions.append(source_matrix)
            frame_decoded_confusions.append(decoded_matrix)
        for rate_id in range(6):
            label = selections[rate_id][item["frame_id"]]
            adaptive_source[rate_id] += frame_source_confusions[label]
            adaptive_decoded[rate_id] += frame_decoded_confusions[label]
            selection_counts[rate_id, label] += 1
        completed_ids.append(item["frame_id"])
        if index == 1 or index % 25 == 0:
            elapsed = time.time() - start
            rate = index / max(elapsed, 1e-6)
            eta = (len(worker_items) - index) / max(rate, 1e-6)
            log(f"{index}/{len(worker_items)} frames; ETA {eta / 60.0:.1f} min")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / f"rd_shard_{rank:02d}_of_{world_size:02d}.npz"
    np.savez_compressed(
        shard_path,
        fixed_source_confusion=fixed_source,
        adaptive_source_confusion=adaptive_source,
        fixed_decoded_confusion=fixed_decoded,
        adaptive_decoded_confusion=adaptive_decoded,
        selection_counts=selection_counts,
        frame_ids=np.asarray(completed_ids, dtype="U6"),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        steps_mm=np.asarray(args.steps_mm, dtype=np.int64),
    )
    log(f"Saved validated RD shard: {shard_path}")


def metrics_from_confusion(matrix):
    matrix = matrix.astype(np.float64, copy=False)
    intersection = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    present = union > 0
    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    iou[present] = intersection[present] / union[present]
    total = matrix.sum()
    return {
        "miou": float(np.nanmean(iou)),
        "accuracy": float(intersection.sum() / total) if total else 0.0,
        "class_iou": iou,
    }


def load_gpcc_table(path, expected_ids):
    table = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = normalize_frame_id(row.get("filename") or row.get("frame_id"))
            label = int(row["rate_id"])
            key = (frame_id, label)
            if key in table:
                raise RuntimeError(f"Duplicate G-PCC measurement {key}")
            table[key] = {
                "bits": int(float(row["bits"])),
                "num_points": int(float(row["num_points"])),
            }
    missing = [
        (frame_id, label)
        for frame_id in expected_ids
        for label in range(6)
        if (frame_id, label) not in table
    ]
    if missing:
        raise RuntimeError(
            f"G-PCC table is missing {len(missing)} rows, first={missing[:3]}"
        )
    return table


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(baseline_rows, adaptive_rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = sorted(baseline_rows, key=lambda row: row["bpp"])
    adaptive = sorted(adaptive_rows, key=lambda row: row["bpp"])
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), dpi=160)
    panels = (
        (axes[0], "decoded_miou", "Decoded-domain mIoU (primary)"),
        (axes[1], "source_miou", "Original-point mIoU (audit)"),
    )
    for axis, metric, title in panels:
        axis.plot(
            [row["bpp"] for row in baseline],
            [100.0 * row[metric] for row in baseline],
            "o--", linewidth=2.0, markersize=6,
            label="Fixed-step G-PCC baseline",
        )
        axis.plot(
            [row["bpp"] for row in adaptive],
            [100.0 * row[metric] for row in adaptive],
            "s-", linewidth=2.2, markersize=6,
            label="Segmentation-proxy guided G-PCC",
        )
        axis.set_xlabel("G-PCC bitrate (bits per original point)")
        axis.set_ylabel("SemanticKITTI sequence-08 mIoU (%)")
        axis.set_title(title)
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.legend(loc="best")
    for row in baseline:
        axes[0].annotate(
            f"{int(row['quant_step_mm'])}mm",
            (row["bpp"], 100.0 * row["decoded_miou"]),
            xytext=(4, -12), textcoords="offset points", fontsize=7.5,
        )
    for row in adaptive:
        axes[0].annotate(
            f"T={row['threshold']:.5g}",
            (row["bpp"], 100.0 * row["decoded_miou"]),
            xytext=(4, 6), textcoords="offset points", fontsize=7.5,
        )
    fig.suptitle("SemanticKITTI decoded-mIoU-loss routing")
    fig.tight_layout()
    png = output_dir / "segmentation_proxy_guided_miou_bpp.png"
    pdf = output_dir / "segmentation_proxy_guided_miou_bpp.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def merge_shards(args, items):
    expected_ids = [item["frame_id"] for item in items]
    manifest, thresholds, selections = load_router_labels(
        args.router_manifest, expected_ids
    )
    fixed_source = np.zeros((6, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    adaptive_source = np.zeros_like(fixed_source)
    fixed_decoded = np.zeros_like(fixed_source)
    adaptive_decoded = np.zeros_like(fixed_source)
    selection_counts = np.zeros((6, 6), dtype=np.int64)
    completed = []
    shard_paths = []
    for rank in range(args.expected_world_size):
        path = args.output_dir / (
            f"rd_shard_{rank:02d}_of_{args.expected_world_size:02d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        shard_paths.append(path)
        with np.load(path) as payload:
            if tuple(payload["steps_mm"].tolist()) != tuple(args.steps_mm):
                raise RuntimeError(f"Step mismatch in {path}")
            if not np.allclose(payload["thresholds"], thresholds):
                raise RuntimeError(f"Threshold mismatch in {path}")
            fixed_source += payload["fixed_source_confusion"]
            adaptive_source += payload["adaptive_source_confusion"]
            fixed_decoded += payload["fixed_decoded_confusion"]
            adaptive_decoded += payload["adaptive_decoded_confusion"]
            selection_counts += payload["selection_counts"]
            completed.extend(payload["frame_ids"].tolist())
    if len(completed) != len(set(completed)):
        raise RuntimeError("Duplicate frame IDs across RD shards")
    if set(completed) != set(expected_ids):
        raise RuntimeError(
            f"RD shard coverage mismatch: complete={len(completed)}, expected={len(expected_ids)}"
        )

    gpcc = load_gpcc_table(args.gpcc_details, expected_ids)
    total_points = sum(gpcc[(frame_id, 0)]["num_points"] for frame_id in expected_ids)
    baseline_rows = []
    for label, step_mm in enumerate(args.steps_mm):
        source_stats = metrics_from_confusion(fixed_source[label])
        decoded_stats = metrics_from_confusion(fixed_decoded[label])
        total_bits = sum(gpcc[(frame_id, label)]["bits"] for frame_id in expected_ids)
        baseline_rows.append(
            {
                "rate_id": label,
                "quant_step_mm": int(step_mm),
                "position_quantization_scale": 1.0 / float(step_mm),
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
        )

    adaptive_rows = []
    for rate_id, threshold in enumerate(thresholds):
        source_stats = metrics_from_confusion(adaptive_source[rate_id])
        decoded_stats = metrics_from_confusion(adaptive_decoded[rate_id])
        total_bits = 0
        for frame_id in expected_ids:
            label = selections[rate_id][frame_id]
            total_bits += gpcc[(frame_id, label)]["bits"]
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
        adaptive_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = args.output_dir / "gpcc_fixed_step_miou_bpp.csv"
    adaptive_csv = args.output_dir / "segmentation_proxy_guided_miou_bpp.csv"
    write_csv(baseline_csv, baseline_rows)
    write_csv(adaptive_csv, adaptive_rows)
    png, pdf = plot_curves(baseline_rows, adaptive_rows, args.output_dir)
    summary = {
        "status": "complete",
        "completed_at": time.strftime("%F %T"),
        "evaluation_split": "SemanticKITTI labeled sequence 08",
        "num_frames": len(expected_ids),
        "primary_metric": (
            "global decoded-domain 19-class mIoU after nearest-neighbour "
            "ground-truth recolouring"),
        "audit_metric": (
            "global original-point 19-class mIoU after nearest-neighbour "
            "prediction transfer"),
        "bpp": "total G-PCC geometry bitstream bits / total original points",
        "nearest_neighbor_transfer": True,
        "quant_steps_mm_coarse_to_fine": list(args.steps_mm),
        "segmentation_loss_thresholds": thresholds,
        "router_manifest": str(args.router_manifest.resolve()),
        "router_checkpoint": manifest.get("ckpt", ""),
        "segmentation_checkpoint": str(args.checkpoint.resolve()),
        "gpcc_details": str(args.gpcc_details.resolve()),
        "baseline_csv": str(baseline_csv.resolve()),
        "adaptive_csv": str(adaptive_csv.resolve()),
        "plot_png": str(png.resolve()),
        "plot_pdf": str(pdf.resolve()),
        "shards": [str(path.resolve()) for path in shard_paths],
    }
    marker = args.output_dir / "SEGMENTATION_PROXY_RD_COMPLETE.json"
    marker.write_text(json.dumps(summary, indent=2))
    log(f"Baseline curve: {baseline_csv}")
    log(f"Adaptive curve: {adaptive_csv}")
    log(f"Plot: {png}")


def validate_args(args):
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if tuple(args.steps_mm) != DEFAULT_STEPS_MM:
        raise ValueError(f"Expected steps {DEFAULT_STEPS_MM}, got {tuple(args.steps_mm)}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if args.prepare_only:
        return
    for name in ("config", "checkpoint", "router_manifest"):
        path = getattr(args, name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing --{name.replace('_', '-')}: {path}")
        setattr(args, name, path.resolve())
    if args.merge_only:
        if args.gpcc_details is None or not args.gpcc_details.is_file():
            raise FileNotFoundError(args.gpcc_details)
        args.gpcc_details = args.gpcc_details.resolve()


def main():
    args = parse_args()
    validate_args(args)
    items = load_val_items(args.dataset_root, args.max_frames)
    if args.prepare_only:
        prepare_proxy_split(args, items)
    elif args.merge_only:
        merge_shards(args, items)
    else:
        evaluate_worker(args, items)


if __name__ == "__main__":
    main()
