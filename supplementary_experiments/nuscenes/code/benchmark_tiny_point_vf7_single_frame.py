#!/usr/bin/env python3
"""Benchmark TinyPoint-VF7 core and GPU-preprocessed single-frame latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gpu_voxelizer import voxelize_batch_gpu
from tiny_point_vf7_absolute_loss_monotonic_rate_proxy import (
    TinyPointVF7AbsoluteLossMonotonicRateProxy,
    count_parameters,
)


def pack(features, coords):
    if int(coords[:, 0].min()) != 0 or int(coords[:, 0].max()) != 0:
        raise RuntimeError("Expected a batch-one voxel tensor")
    points = features[None, ...]
    mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    return points, mask


def synchronize_time(callable_, warmup, iterations):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        values.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(values, np.float64)
    return {
        "iterations": int(iterations),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--point-width", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["model"].items()
    }
    train_args = checkpoint["args"]
    loss_scales = state["loss_scales"].float().numpy()
    mean_log_bpp = torch.cumsum(state["mean_log_increments"].float(), 0).numpy()
    device = torch.device("cuda:0")
    model = TinyPointVF7AbsoluteLossMonotonicRateProxy(
        int(train_args.get("feat_dim", 256)), loss_scales, mean_log_bpp
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    raw = np.fromfile(args.points, dtype=np.float32)
    if raw.size % args.point_width:
        raise ValueError(f"Invalid point file shape: {args.points}")
    xyz = torch.from_numpy(
        raw.reshape(-1, args.point_width)[:, :3].copy()
    ).to(device)
    voxel_size = train_args.get("voxel_size", [0.16, 0.16, 0.16])
    pc_range = train_args.get(
        "point_cloud_range", [-100, -100, -20, 100, 100, 20]
    )
    max_voxels = int(train_args.get("max_voxels", 60000))

    def preprocess():
        features, coords = voxelize_batch_gpu(
            [xyz], voxel_size, pc_range, max_voxels,
            use_abs_xyz=True, include_intensity=False, random_subsample=False,
        )
        return pack(features, coords)

    packed, valid_mask = preprocess()
    if packed.shape[-1] != 7:
        raise RuntimeError(f"Expected 7 features, got {packed.shape[-1]}")

    with torch.inference_mode():
        core = synchronize_time(
            lambda: model(packed, valid_mask), args.warmup, args.iterations
        )

        def end_to_end():
            current, mask = preprocess()
            return model(current, mask)

        torch.cuda.reset_peak_memory_stats()
        end = synchronize_time(end_to_end, args.warmup, args.iterations)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    result = {
        "model_alias": "TinyPoint-VF7",
        "point_file": str(args.points.resolve()),
        "raw_points": int(xyz.shape[0]),
        "active_voxels": int(packed.shape[1]),
        "voxel_size": list(map(float, voxel_size)),
        "parameters": count_parameters(model),
        "protocol": "batch=1; CUDA synchronize each iteration; file I/O excluded",
        "core_network": core,
        "gpu_voxelization_pack_and_network": end,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
