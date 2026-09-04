#!/usr/bin/env python3
"""One-frame complexity benchmark for SUN RGB-D TinyPoint and VoteNet."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


MIB = 1024.0 ** 2


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def benchmark_tiny(args):
    sys.path.insert(0, str(args.code_dir.resolve()))
    from tiny_point_absolute_loss_monotonic_rate_proxy import (
        TinyPointAbsoluteLossMonotonicRateProxy,
        count_parameters,
    )

    offsets = np.load(args.cache_dir / "val_offsets_160mm.npy", mmap_mode="r")
    flat = np.load(args.cache_dir / "val_points_160mm.npy", mmap_mode="r")
    scene_ids = np.load(args.cache_dir / "val_scene_ids.npy", allow_pickle=False)
    start, end = int(offsets[args.index]), int(offsets[args.index + 1])
    xyz = np.ascontiguousarray(np.asarray(flat[start:end]), dtype=np.float32).copy()
    checkpoint = torch.load(args.router_checkpoint, map_location="cpu")
    model = TinyPointAbsoluteLossMonotonicRateProxy(
        256, checkpoint["loss_scales"], checkpoint["mean_log_bpp"], input_channels=3
    )
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    counts = count_parameters(model)
    point_cloud_range = checkpoint["args"]["point_cloud_range"]
    lower = torch.tensor(point_cloud_range[:3], device="cuda", dtype=torch.float32)
    upper = torch.tensor(point_cloud_range[3:], device="cuda", dtype=torch.float32)

    def prepare(array):
        points = torch.from_numpy(array[None]).cuda()
        points = ((points - lower) / (upper - lower + 1e-6)) * 2.0 - 1.0
        mask = torch.ones(points.shape[:2], device="cuda", dtype=torch.bool)
        return points, mask

    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    base_points, base_mask = prepare(xyz)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(base_points, base_mask)
    torch.cuda.synchronize()

    core_times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            started = time.perf_counter()
            model(base_points, base_mask)
            torch.cuda.synchronize()
            core_times.append((time.perf_counter() - started) * 1000.0)
    core_memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "incremental_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ) / MIB,
    }

    e2e_times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            started = time.perf_counter()
            current_points, current_mask = prepare(xyz.copy())
            model(current_points, current_mask)
            torch.cuda.synchronize()
            e2e_times.append((time.perf_counter() - started) * 1000.0)
            del current_points, current_mask
    e2e_memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "incremental_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ) / MIB,
    }
    result = {
        "component": "TinyPoint router (160 mm cell means)",
        "dataset": "SUN RGB-D validation",
        "sample_id": str(scene_ids[args.index]),
        "input": {"points": len(xyz), "xyz_fp32_bytes": int(xyz.nbytes)},
        "model": {
            "parameters": counts["total"],
            "trainable_parameters": counts["trainable"],
            "parameter_breakdown": counts,
            "fp32_weight_mib": counts["total"] * 4 / MIB,
        },
        "latency": {
            "core_network": stats(core_times),
            "end_to_end_in_memory": stats(e2e_times),
        },
        "memory": {
            "baseline_allocated_mib": baseline_allocated / MIB,
            "baseline_reserved_mib": baseline_reserved / MIB,
            "core": core_memory,
            "end_to_end": e2e_memory,
        },
        "protocol": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "disk_io_excluded": True,
            "end_to_end_scope": "host-to-device, fixed-range normalization, TinyPoint forward",
            "offline_160mm_cell_construction_excluded": True,
        },
    }
    del model, base_points, base_mask, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_votenet(args):
    sys.path.insert(0, str(args.mmdet_root.resolve()))
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import DATASETS, MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    config = Config.fromfile(str(args.votenet_config.resolve()))
    dataset = DATASETS.build(config.test_dataloader.dataset)
    dataset.full_init()
    sample = dataset[args.index]
    point_count = int(sample["inputs"]["points"].shape[0])
    model = MODELS.build(config.model)
    load_checkpoint(model, str(args.votenet_checkpoint.resolve()), map_location="cpu")
    model.cuda().eval()
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    processed = model.data_preprocessor(pseudo_collate([sample]), training=False)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**processed, mode="predict")
    torch.cuda.synchronize()

    core_times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            started = time.perf_counter()
            model(**processed, mode="predict")
            torch.cuda.synchronize()
            core_times.append((time.perf_counter() - started) * 1000.0)
    core_memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "incremental_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ) / MIB,
    }

    cpu_times, preprocess_times, predict_times, e2e_times = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            total_started = time.perf_counter()
            current_sample = dataset[args.index]
            cpu_done = time.perf_counter()
            current = model.data_preprocessor(pseudo_collate([current_sample]), training=False)
            torch.cuda.synchronize()
            preprocess_done = time.perf_counter()
            model(**current, mode="predict")
            torch.cuda.synchronize()
            done = time.perf_counter()
            cpu_times.append((cpu_done - total_started) * 1000.0)
            preprocess_times.append((preprocess_done - cpu_done) * 1000.0)
            predict_times.append((done - preprocess_done) * 1000.0)
            e2e_times.append((done - total_started) * 1000.0)
    e2e_memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "incremental_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ) / MIB,
    }
    return {
        "component": "VoteNet geometry-only detector",
        "dataset": "SUN RGB-D validation",
        "sample_id": str(args.index),
        "input": {"sampled_points": point_count},
        "model": {
            "parameters": parameters,
            "trainable_parameters": int(sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )),
            "fp32_weight_mib": parameters * 4 / MIB,
        },
        "latency": {
            "core_network": stats(core_times),
            "cpu_dataset_pipeline_including_cached_file_read": stats(cpu_times),
            "device_preprocessor": stats(preprocess_times),
            "predict_with_postprocess": stats(predict_times),
            "end_to_end_framework_predict": stats(e2e_times),
        },
        "memory": {
            "baseline_allocated_mib": baseline_allocated / MIB,
            "baseline_reserved_mib": baseline_reserved / MIB,
            "core": core_memory,
            "end_to_end": e2e_memory,
        },
        "protocol": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "disk_io_excluded_from_core": True,
            "end_to_end_scope": (
                "official dataset pipeline and cached file read, device preprocessing, "
                "VoteNet prediction and postprocess"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--mmdet-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--router-checkpoint", required=True, type=Path)
    parser.add_argument("--votenet-config", required=True, type=Path)
    parser.add_argument("--votenet-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiny = benchmark_tiny(args)
    (args.output_dir / "sunrgbd_tiny_point.json").write_text(json.dumps(tiny, indent=2))
    votenet = benchmark_votenet(args)
    (args.output_dir / "sunrgbd_votenet.json").write_text(json.dumps(votenet, indent=2))
    summary = {
        "tiny_point": {
            "parameters": tiny["model"]["parameters"],
            "e2e_p50_ms": tiny["latency"]["end_to_end_in_memory"]["p50_ms"],
            "e2e_incremental_peak_mib": tiny["memory"]["end_to_end"]["incremental_peak_allocated_mib"],
        },
        "votenet": {
            "parameters": votenet["model"]["parameters"],
            "e2e_p50_ms": votenet["latency"]["end_to_end_framework_predict"]["p50_ms"],
            "e2e_incremental_peak_mib": votenet["memory"]["end_to_end"]["incremental_peak_allocated_mib"],
        },
    }
    (args.output_dir / "SUNRGBD_COMPLEXITY_COMPLETE.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
