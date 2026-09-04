#!/usr/bin/env python3
"""Select per-frame quantization labels from direct PV-RCNN loss increases.

The legacy router exported a learned AP-drop proxy. This replacement evaluates
the detector itself in its training/loss branch for each fixed quantization
combo. Loss deltas are measured against the finest selectable quantization
combo, not against the raw unquantized point cloud.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPCDET_TOOLS = PROJECT_ROOT / "OpenPCDet" / "tools"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OPENPCDET_TOOLS))

from integrations.openpcdet import install_openpcdet_compat

install_openpcdet_compat()

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_scale(value):
    value = str(value).strip()
    if not value:
        raise ValueError("empty quantization scale")
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator = float(denominator)
        if denominator == 0:
            raise ValueError(f"invalid scale: {value}")
        return float(numerator) / denominator
    return float(value)


def parse_quant_map(text):
    combos = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid quant map item {item!r}; expected fg_scale,bg_scale"
            )
        fg, bg = parse_scale(parts[0]), parse_scale(parts[1])
        if fg <= 0 or bg <= 0:
            raise ValueError(
                "Direct detector-loss evaluation requires positive scales; "
                f"got {item!r}"
            )
        combos.append((fg, bg))
    if not combos:
        raise ValueError("--quant_map is empty")
    return combos


def parse_int_list(text, name):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def parse_float_list(text, name):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"{name} must contain at least one number")
    return values


def norm_frame_id(value):
    return str(value).strip().zfill(6)


def read_split_file(path):
    with open(path) as handle:
        return [norm_frame_id(line) for line in handle if line.strip()]


def quantize_subset(coords_scaled, mask, scale):
    subset = coords_scaled[mask]
    if len(subset) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if scale >= 1.0:
        return subset.astype(np.float64)
    quantized = np.round(subset.astype(np.float64) * scale).astype(np.int32)
    unique_quantized = np.unique(quantized, axis=0)
    return unique_quantized.astype(np.float64) / scale


def quantize_points(points, scale_fg, scale_bg, seg_labels=None):
    """Match test_split.py's millimetre-origin quantization exactly."""
    if len(points) == 0:
        return points.astype(np.float32, copy=True)

    coords_raw = points[:, :3]
    coords_mm = np.round(coords_raw.astype(np.float64) * 1000.0).astype(np.int32)
    offset = coords_mm.min(axis=0)
    coords_scaled = coords_mm - offset

    if seg_labels is None:
        if not np.isclose(scale_fg, scale_bg):
            raise ValueError(
                "Without --mask_dir each combo must use the same fg/bg scale; "
                f"got {scale_fg} and {scale_bg}"
            )
        all_mask = np.ones(len(points), dtype=bool)
        decoded = quantize_subset(coords_scaled, all_mask, scale_fg)
    else:
        seg_labels = np.asarray(seg_labels).reshape(-1)
        if len(seg_labels) != len(points):
            raise ValueError(
                f"Mask length {len(seg_labels)} does not match raw point count {len(points)}"
            )
        fg_mask = seg_labels == 1
        bg_mask = ~fg_mask
        decoded_fg = quantize_subset(coords_scaled, fg_mask, scale_fg)
        decoded_bg = quantize_subset(coords_scaled, bg_mask, scale_bg)
        if len(decoded_fg) and len(decoded_bg):
            decoded = np.concatenate([decoded_fg, decoded_bg], axis=0)
        elif len(decoded_fg):
            decoded = decoded_fg
        else:
            decoded = decoded_bg

    decoded = (decoded + offset.astype(np.float64)) / 1000.0
    zeros = np.zeros((len(decoded), 1), dtype=np.float32)
    return np.concatenate([decoded.astype(np.float32), zeros], axis=1)


def install_quantized_get_lidar(dataset, mask_dir):
    """Install a single-process dataset hook used by dataset.__getitem__."""
    original_get_lidar = dataset.__class__.get_lidar
    dataset.__class__.loss_quantization = None
    mask_dir = Path(mask_dir).resolve() if mask_dir else None

    def get_lidar_with_quantization(self, idx):
        points = original_get_lidar(self, idx)
        current = getattr(self.__class__, "loss_quantization", None)
        if current is None:
            return points

        scale_fg, scale_bg = current
        seg_labels = None
        if mask_dir is not None:
            frame_id = norm_frame_id(idx)
            mask_path = mask_dir / f"{frame_id}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing segmentation mask: {mask_path}")
            seg_labels = np.load(mask_path)
        return quantize_points(points, scale_fg, scale_bg, seg_labels)

    dataset.__class__.get_lidar = get_lidar_with_quantization


def set_loss_mode(model):
    """Enable PV-RCNN's loss branch without stochastic dropout/BN updates."""
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            module.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def evaluate_one_loss(model, dataset, dataset_index):
    data_dict = dataset[dataset_index]
    batch_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch_dict)
    with torch.no_grad():
        ret_dict, tb_dict, _ = model(batch_dict)
    loss = ret_dict["loss"].detach().float().item()
    stats = {key: float(value) for key, value in tb_dict.items()}
    stats["total_loss"] = loss
    return stats


def scalar_or_zero(value):
    return round(float(value), 8)


def sanitized_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (Path(__file__).resolve().parent / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Select quantization labels from direct PV-RCNN loss increases."
    )
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument(
        "--quant_map",
        required=True,
        help="fg,bg;fg,bg;... quantization combinations; selectable labels are six entries",
    )
    parser.add_argument(
        "--candidate_labels",
        default=None,
        help="Actual combo labels to select; default is 1..6 for 7 combos or 0..5 for 6 combos",
    )
    parser.add_argument(
        "--loss_thresholds",
        required=True,
        help="Six nondecreasing absolute loss tolerances, comma separated",
    )
    parser.add_argument("--mask_dir", default=None, help="Optional split foreground/background mask directory")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="detection_loss")
    parser.add_argument("--loss_csv", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    launch_dir = Path.cwd().resolve()
    tools_dir = Path(__file__).resolve().parent
    cfg_path = resolve_path(args.cfg_file, launch_dir)
    ckpt_path = resolve_path(args.ckpt, launch_dir)
    split_path = resolve_path(args.split_file, launch_dir)
    mask_dir = resolve_path(args.mask_dir, launch_dir) if args.mask_dir else None
    out_dir = resolve_path(args.out_dir, launch_dir)
    loss_csv = resolve_path(args.loss_csv, launch_dir) if args.loss_csv else out_dir / f"{args.prefix}_sensitivity.csv"

    quant_map = parse_quant_map(args.quant_map)
    if args.candidate_labels is None:
        candidate_labels = list(range(1, len(quant_map))) if len(quant_map) == 7 else list(range(len(quant_map)))
    else:
        candidate_labels = parse_int_list(args.candidate_labels, "--candidate_labels")
    thresholds = parse_float_list(args.loss_thresholds, "--loss_thresholds")

    if len(candidate_labels) != 6:
        raise ValueError(f"Expected exactly six candidate labels, got {candidate_labels}")
    if len(thresholds) != 6:
        raise ValueError(f"Expected exactly six loss thresholds, got {thresholds}")
    if any(label < 0 or label >= len(quant_map) for label in candidate_labels):
        raise ValueError(f"candidate_labels {candidate_labels} exceed quant_map length {len(quant_map)}")
    if candidate_labels != sorted(candidate_labels) or len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("candidate_labels must be unique and numerically ordered")
    if any(cur < prev for prev, cur in zip(thresholds, thresholds[1:])):
        raise ValueError("loss thresholds must be nondecreasing")
    if args.mask_dir and not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing --mask_dir: {mask_dir}")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Direct PV-RCNN loss evaluation requires a CUDA device")

    # OpenPCDet resolves _BASE_CONFIG_ paths relative to OpenPCDet/tools.
    os.chdir(tools_dir)
    cfg_from_yaml_file(str(cfg_path), cfg)
    logger = common_utils.create_logger()

    print("[direct-loss] Building KITTI dataset and PV-RCNN model...")
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    install_quantized_get_lidar(dataset, mask_dir)

    dataset_frame_to_index = {
        norm_frame_id(info["point_cloud"]["lidar_idx"]): index
        for index, info in enumerate(dataset.kitti_infos)
    }
    requested_frame_ids = read_split_file(split_path)
    missing = [frame_id for frame_id in requested_frame_ids if frame_id not in dataset_frame_to_index]
    if missing:
        raise KeyError(f"{len(missing)} split frames are missing from dataset info, first={missing[:5]}")
    frame_indices = [dataset_frame_to_index[frame_id] for frame_id in requested_frame_ids]
    start = max(0, int(args.start_index))
    stop = len(frame_indices) if args.max_frames is None else min(len(frame_indices), start + int(args.max_frames))
    frame_indices = frame_indices[start:stop]
    frame_ids = requested_frame_ids[start:stop]
    if not frame_indices:
        raise ValueError("No frames selected")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.cuda()
    set_loss_mode(model)

    # In this quantization implementation, the scale is the inverse spatial
    # grid step. A larger scale therefore represents the finer quantization.
    finest_label = max(
        candidate_labels,
        key=lambda label: (quant_map[label][0], quant_map[label][1]),
    )
    # For a threshold, choose the coarsest (lowest-scale) valid candidate to
    # maximize compression. The current six candidates are ordered coarse to
    # fine, but sorting by scale keeps this rule explicit and robust.
    coarse_to_fine_labels = sorted(
        candidate_labels,
        key=lambda label: (quant_map[label][0], quant_map[label][1]),
    )

    print(
        "[direct-loss] LOSS_CALCULATION_BEGIN "
        f"frames={len(frame_indices)} candidates={candidate_labels} "
        f"finest_label={finest_label} "
        f"thresholds={','.join(str(value) for value in thresholds)}"
    )
    print(
        "[direct-loss] Baseline is the finest quantized candidate "
        f"L{finest_label}; deltas are Lq-Lfinest."
    )

    loss_rows = []
    label_rows = [[] for _ in thresholds]
    total_start = __import__("time").time()
    for frame_id, dataset_index in tqdm(
        list(zip(frame_ids, frame_indices)), desc="direct detector loss", dynamic_ncols=True
    ):
        candidate_stats = {}
        for label in candidate_labels:
            dataset.__class__.loss_quantization = quant_map[label]
            stats = evaluate_one_loss(model, dataset, dataset_index)
            candidate_stats[label] = stats

        finest_stats = candidate_stats[finest_label]
        finest_loss = finest_stats["total_loss"]
        for label in candidate_labels:
            stats = candidate_stats[label]
            signed_delta = stats["total_loss"] - finest_loss
            stats["signed_delta"] = signed_delta
            stats["loss_delta"] = signed_delta

        dataset.__class__.loss_quantization = None
        loss_row = {
            "frame_id": frame_id,
            "finest_label": finest_label,
            "finest_total_loss": scalar_or_zero(finest_loss),
        }
        for key, value in finest_stats.items():
            if key not in {"total_loss", "signed_delta", "loss_delta"}:
                loss_row[f"finest_{key}"] = scalar_or_zero(value)
        for label in candidate_labels:
            stats = candidate_stats[label]
            loss_row[f"L{label}_total_loss"] = scalar_or_zero(stats["total_loss"])
            loss_row[f"L{label}_loss_delta"] = scalar_or_zero(stats["loss_delta"])
            loss_row[f"L{label}_signed_delta"] = scalar_or_zero(stats["signed_delta"])
            for key, value in stats.items():
                if key not in {"total_loss", "signed_delta", "loss_delta"}:
                    loss_row[f"L{label}_{key}"] = scalar_or_zero(value)

        for rate_id, threshold in enumerate(thresholds):
            valid_labels = [
                label
                for label in coarse_to_fine_labels
                if candidate_stats[label]["loss_delta"] <= threshold
            ]
            chosen_label = valid_labels[0] if valid_labels else finest_label
            chosen_stats = candidate_stats[chosen_label]
            fg, bg = quant_map[chosen_label]
            label_rows[rate_id].append(
                {
                    "frame_id": frame_id,
                    "jucp_label": chosen_label,
                    "rate_id": rate_id,
                    "threshold": scalar_or_zero(threshold),
                    "finest_label": finest_label,
                    "finest_total_loss": scalar_or_zero(finest_loss),
                    "selected_total_loss": scalar_or_zero(chosen_stats["total_loss"]),
                    "selected_loss_delta": scalar_or_zero(chosen_stats["loss_delta"]),
                    "selected_signed_delta": scalar_or_zero(chosen_stats["signed_delta"]),
                    "posQ_fg": scalar_or_zero(fg),
                    "posQ_bg": scalar_or_zero(bg),
                    "quant_step_fg_mm": scalar_or_zero(1.0 / fg),
                    "quant_step_bg_mm": scalar_or_zero(1.0 / bg),
                }
            )
        loss_rows.append(loss_row)

    write_csv(loss_csv, loss_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "direct_pv_rcnn_loss",
        "cfg_file": str(cfg_path),
        "ckpt": str(ckpt_path),
        "split_file": str(split_path),
        "mask_dir": str(mask_dir or ""),
        "quant_map": args.quant_map,
        "candidate_labels": candidate_labels,
        "loss_thresholds": thresholds,
        "baseline": "finest selectable quantized candidate",
        "baseline_label": finest_label,
        "baseline_quantization": {
            "posQ_fg": scalar_or_zero(quant_map[finest_label][0]),
            "posQ_bg": scalar_or_zero(quant_map[finest_label][1]),
            "quant_step_fg_mm": scalar_or_zero(1.0 / quant_map[finest_label][0]),
            "quant_step_bg_mm": scalar_or_zero(1.0 / quant_map[finest_label][1]),
        },
        "loss_definition": "candidate_total_loss - finest_quantized_total_loss",
        "loss_csv": str(loss_csv),
        "label_csvs": [],
        "num_frames": len(loss_rows),
        "elapsed_seconds": __import__("time").time() - total_start,
    }
    for rate_id, rows in enumerate(label_rows):
        threshold = thresholds[rate_id]
        out_csv = out_dir / f"{args.prefix}_rate_{rate_id}_{sanitized_number(threshold)}.csv"
        write_csv(out_csv, rows)
        manifest["label_csvs"].append(
            {"rate_id": rate_id, "threshold": scalar_or_zero(threshold), "path": str(out_csv)}
        )
    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[direct-loss] Loss CSV: {loss_csv}")
    print(f"[direct-loss] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
