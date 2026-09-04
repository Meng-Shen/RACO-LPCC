#!/usr/bin/env python3
"""Export per-frame PV-RCNN losses for a hybrid G-PCC/residual candidate set.

Rates 0..4 are read from already restored RENO point clouds.  Rate 5 is the
plain 64 mm G-PCC geometry, reconstructed deterministically from the source
cloud without running the codec again.  Actual G-PCC bits are handled by the
oracle aggregation script and are never estimated here.
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
import torch.nn as nn
from tqdm import tqdm


ROOT = Path("/public/DATA/sm/RACO-LPCC")
TOOLS = ROOT / "OpenPCDet" / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from integrations.openpcdet import install_openpcdet_compat  # noqa: E402

install_openpcdet_compat()
import _init_path  # noqa: F401,E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def norm_frame_id(value: object) -> str:
    return str(value).strip().zfill(6)


def read_ids(path: Path) -> list[str]:
    frame_ids = [norm_frame_id(line) for line in path.read_text().splitlines() if line.strip()]
    if not frame_ids:
        raise ValueError(f"no frame ids found in {path}")
    return frame_ids


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values:
        raise ValueError("integer list is empty")
    return values


def plain_quantized_points(points: np.ndarray, q_step_mm: int) -> np.ndarray:
    """Match the fixed whole-frame G-PCC geometry used by this project."""

    xyz_mm = np.rint(points[:, :3].astype(np.float64) * 1000.0).astype(np.int64)
    origin_mm = xyz_mm.min(axis=0)
    anchors = np.rint(
        (xyz_mm - origin_mm[None, :]).astype(np.float64) / float(q_step_mm)
    ).astype(np.int64)
    anchors = np.unique(anchors, axis=0)
    decoded_xyz = (origin_mm[None, :] + anchors * int(q_step_mm)).astype(np.float32) * 0.001
    decoded = np.zeros((len(decoded_xyz), 4), dtype=np.float32)
    decoded[:, :3] = decoded_xyz
    return np.ascontiguousarray(decoded)


def set_loss_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            module.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def evaluate_one(model: nn.Module, dataset, dataset_index: int) -> dict[str, float]:
    data_dict = dataset[dataset_index]
    batch_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch_dict)
    with torch.no_grad():
        ret_dict, tb_dict, _ = model(batch_dict)
    stats = {key: float(value) for key, value in tb_dict.items()}
    stats["total_loss"] = float(ret_dict["loss"].detach().float().item())
    return stats


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_reused_plain_losses(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    reused = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = norm_frame_id(row["frame_id"])
            reused[frame_id] = float(row["L5_total_loss"])
    return reused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "val"),
        default="val",
        help="OpenPCDet split whose info PKL and sample list are loaded.",
    )
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--decoded-rate-ids", default="0,1,2,3,4")
    parser.add_argument("--plain-rate-id", type=int, default=5)
    parser.add_argument("--plain-q-step-mm", type=int, default=64)
    parser.add_argument(
        "--reuse-plain-loss-csv",
        type=Path,
        default=None,
        help="Existing same-detector six-scale labels; reuse L5_total_loss and skip 64 mm forward.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    decoded_rates = parse_ints(args.decoded_rate_ids)
    all_rates = sorted(decoded_rates + [args.plain_rate_id])
    if all_rates != list(range(6)):
        raise ValueError(f"expected the six rates 0..5, got {all_rates}")
    for rate_id in decoded_rates:
        rate_dir = args.decoded_dir / f"rate_{rate_id}"
        if not rate_dir.is_dir():
            raise FileNotFoundError(rate_dir)

    os.chdir(TOOLS)
    cfg_from_yaml_file(str(args.cfg_file.resolve()), cfg)
    # build_dataloader(training=False) indexes DATA_SPLIT/INFO_PATH through the
    # ``test`` key.  Explicitly redirect both fields for train-label export so
    # training frames are evaluated without enabling data augmentation.
    if args.dataset_split == "train":
        cfg.DATA_CONFIG.DATA_SPLIT["test"] = cfg.DATA_CONFIG.DATA_SPLIT["train"]
        cfg.DATA_CONFIG.INFO_PATH["test"] = list(cfg.DATA_CONFIG.INFO_PATH["train"])
    logger = common_utils.create_logger(rank=0)
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    frame_to_index = {
        norm_frame_id(info["point_cloud"]["lidar_idx"]): index
        for index, info in enumerate(dataset.kitti_infos)
    }
    requested = read_ids(args.split_file)
    missing = [frame_id for frame_id in requested if frame_id not in frame_to_index]
    if missing:
        raise KeyError(f"split contains unknown frames, first={missing[:5]}")
    shard_ids = requested[args.shard_id :: args.num_shards]
    if args.max_frames > 0:
        shard_ids = shard_ids[: args.max_frames]

    original_get_lidar = dataset.__class__.get_lidar
    dataset.__class__.hybrid_rate_id = None

    def hybrid_get_lidar(self, idx):
        rate_id = getattr(self.__class__, "hybrid_rate_id", None)
        if rate_id is None:
            return original_get_lidar(self, idx)
        frame_id = norm_frame_id(idx)
        if rate_id == args.plain_rate_id:
            return plain_quantized_points(original_get_lidar(self, idx), args.plain_q_step_mm)
        decoded_file = args.decoded_dir / f"rate_{rate_id}" / f"{frame_id}.bin"
        if not decoded_file.is_file():
            raise FileNotFoundError(decoded_file)
        values = np.fromfile(decoded_file, dtype=np.float32)
        if values.size == 0 or values.size % 4:
            raise ValueError(f"invalid decoded point cloud: {decoded_file}")
        points = values.reshape(-1, 4)
        points[:, 3] = 0.0
        return np.ascontiguousarray(points)

    dataset.__class__.get_lidar = hybrid_get_lidar
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt.resolve()), logger=logger, to_cpu=False)
    model.cuda()
    set_loss_mode(model)

    reused_plain = load_reused_plain_losses(args.reuse_plain_loss_csv)
    if args.reuse_plain_loss_csv is not None:
        missing_plain = [frame_id for frame_id in requested if frame_id not in reused_plain]
        if missing_plain:
            raise KeyError(f"reuse CSV misses plain 64 mm losses: {missing_plain[:5]}")

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for frame_id in tqdm(shard_ids, desc=f"loss shard {args.shard_id}", dynamic_ncols=True):
        candidate_stats: dict[int, dict[str, float]] = {}
        dataset_index = frame_to_index[frame_id]
        for rate_id in decoded_rates:
            dataset.__class__.hybrid_rate_id = rate_id
            candidate_stats[rate_id] = evaluate_one(model, dataset, dataset_index)
        if reused_plain:
            candidate_stats[args.plain_rate_id] = {
                "total_loss": reused_plain[frame_id]
            }
        else:
            dataset.__class__.hybrid_rate_id = args.plain_rate_id
            candidate_stats[args.plain_rate_id] = evaluate_one(
                model, dataset, dataset_index
            )
        baseline_loss = candidate_stats[args.plain_rate_id]["total_loss"]
        row: dict[str, object] = {
            "frame_id": frame_id,
            "baseline_rate_id": args.plain_rate_id,
            "baseline_total_loss": round(baseline_loss, 8),
        }
        for rate_id in all_rates:
            stats = candidate_stats[rate_id]
            row[f"L{rate_id}_total_loss"] = round(stats["total_loss"], 8)
            row[f"L{rate_id}_signed_delta"] = round(stats["total_loss"] - baseline_loss, 8)
            for key, value in stats.items():
                if key != "total_loss":
                    row[f"L{rate_id}_{key}"] = round(value, 8)
        rows.append(row)
    dataset.__class__.hybrid_rate_id = None

    write_rows(args.output_csv, rows)
    manifest = {
        "mode": "hybrid_gpcc_residual_direct_pv_rcnn_loss",
        "cfg_file": str(args.cfg_file.resolve()),
        "ckpt": str(args.ckpt.resolve()),
        "dataset_split": args.dataset_split,
        "split_file": str(args.split_file.resolve()),
        "decoded_dir": str(args.decoded_dir.resolve()),
        "decoded_rate_ids": decoded_rates,
        "plain_rate_id": args.plain_rate_id,
        "plain_q_step_mm": args.plain_q_step_mm,
        "reused_plain_loss_csv": (
            str(args.reuse_plain_loss_csv.resolve())
            if args.reuse_plain_loss_csv is not None
            else None
        ),
        "plain_rate_forward_recomputed": args.reuse_plain_loss_csv is None,
        "loss_definition": "absolute PV-RCNN total loss; signed deltas use plain 64 mm G-PCC as reference",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_frames": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "output_csv": str(args.output_csv.resolve()),
    }
    args.output_csv.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(
        f"COMPLETE shard={args.shard_id}/{args.num_shards} frames={len(rows)} "
        f"elapsed_s={manifest['elapsed_seconds']:.2f} output={args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
