#!/usr/bin/env python3
"""Benchmark batch-1 current-q all-one coordinate restoration on one KITTI frame."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchsparse.nn import functional as sparse_functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTORER_ROOT = PROJECT_ROOT / "reno" / "current_q_ones_coordinate_v1_20260831"
if str(RESTORER_ROOT) not in sys.path:
    sys.path.insert(0, str(RESTORER_ROOT))

from coordinate_residual import (
    N_BY_Q,
    N_MAX,
    CoordinateResidualNet,
    assert_anchor_alignment,
    build_residual_batch,
    decoded_point_clouds,
)
from gpcc_current_q_ones_coordinate_restore import load_xyz, simulate_gpcc_decoded_xyz


Q_STEPS_MM = (2048, 1024, 512, 256, 128)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--core-repeats", type=int, default=100)
    parser.add_argument("--e2e-repeats", type=int, default=50)
    return parser.parse_args()


def configure_torchsparse():
    config = sparse_functional.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    sparse_functional.conv_config.set_global_conv_config(config)


def latency_stats(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
    }


def timed_call(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def mib(value):
    return float(value) / (1024.0 * 1024.0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    configure_torchsparse()

    model = CoordinateResidualNet(channels=32, kernel_size=3, n_max=N_MAX).to(device)
    model.load_coordinate_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    raw_xyz = load_xyz(args.frame)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    torch.cuda.empty_cache()
    model_resident_allocated = torch.cuda.memory_allocated(device)
    model_resident_reserved = torch.cuda.memory_reserved(device)

    result = {
        "protocol": {
            "frame": str(args.frame),
            "input_points": int(len(raw_xyz)),
            "batch_size": 1,
            "warmup": args.warmup,
            "core_repeats": args.core_repeats,
            "end_to_end_repeats": args.e2e_repeats,
            "core_scope": "prebuilt sparse input -> network XYZ residual predictions",
            "end_to_end_scope": "CPU quantization/dedup + GPU sparse input build + network + GPU coordinate decode; excludes disk I/O",
            "synchronization": "torch.cuda.synchronize before and after each sample",
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "parameters": model_parameters,
            "trainable_parameters": trainable_parameters,
            "fp32_weight_mib": model_parameters * 4.0 / (1024.0 * 1024.0),
            "checkpoint_mib": args.checkpoint.stat().st_size / (1024.0 * 1024.0),
            "resident_allocated_mib": mib(model_resident_allocated),
            "resident_reserved_mib": mib(model_resident_reserved),
        },
        "scales": [],
    }

    with torch.inference_mode():
        for q_step_mm in Q_STEPS_MM:
            n = N_BY_Q[q_step_mm]
            gpcc_xyz = simulate_gpcc_decoded_xyz(raw_xyz, q_step_mm)
            batch = build_residual_batch([gpcc_xyz], q_step_mm, n, device)

            def core_forward():
                return model(batch.input_coords, batch.input_features, N_MAX, q_step_mm)

            for _ in range(args.warmup):
                pred_all, anchor_coords = core_forward()
                assert_anchor_alignment(anchor_coords, batch.anchor_coords)
            torch.cuda.synchronize()

            core_samples = []
            for _ in range(args.core_repeats):
                (_, _), elapsed_ms = timed_call(core_forward)
                core_samples.append(elapsed_ms)

            torch.cuda.empty_cache()
            core_base_allocated = torch.cuda.memory_allocated(device)
            core_base_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            pred_all, anchor_coords = core_forward()
            torch.cuda.synchronize()
            core_peak_allocated = torch.cuda.max_memory_allocated(device)
            core_peak_reserved = torch.cuda.max_memory_reserved(device)
            del pred_all, anchor_coords

            def end_to_end_forward():
                decoded = simulate_gpcc_decoded_xyz(raw_xyz, q_step_mm)
                current_batch = build_residual_batch([decoded], q_step_mm, n, device)
                current_pred, current_anchors = model(
                    current_batch.input_coords,
                    current_batch.input_features,
                    N_MAX,
                    q_step_mm,
                )
                assert_anchor_alignment(current_anchors, current_batch.anchor_coords)
                restored = decoded_point_clouds(
                    current_pred[:, :n],
                    current_anchors,
                    current_batch.origins_mm,
                    q_step_mm,
                )[0]
                return restored, current_batch

            e2e_samples = []
            for _ in range(args.e2e_repeats):
                (restored, current_batch), elapsed_ms = timed_call(end_to_end_forward)
                e2e_samples.append(elapsed_ms)
                del restored, current_batch

            torch.cuda.empty_cache()
            e2e_base_allocated = torch.cuda.memory_allocated(device)
            e2e_base_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            restored, current_batch = end_to_end_forward()
            torch.cuda.synchronize()
            e2e_peak_allocated = torch.cuda.max_memory_allocated(device)
            e2e_peak_reserved = torch.cuda.max_memory_reserved(device)
            restored_points = int(restored.shape[0])
            del restored, current_batch

            row = {
                "q_step_mm": q_step_mm,
                "n_children": n,
                "input_active_voxels": int(batch.input_coords.shape[0]),
                "restored_points": restored_points,
                "core_latency": latency_stats(core_samples),
                "end_to_end_latency": latency_stats(e2e_samples),
                "core_memory": {
                    "base_allocated_mib": mib(core_base_allocated),
                    "base_reserved_mib": mib(core_base_reserved),
                    "peak_allocated_mib": mib(core_peak_allocated),
                    "peak_reserved_mib": mib(core_peak_reserved),
                    "incremental_peak_allocated_mib": mib(
                        core_peak_allocated - core_base_allocated
                    ),
                },
                "end_to_end_memory": {
                    "base_allocated_mib": mib(e2e_base_allocated),
                    "base_reserved_mib": mib(e2e_base_reserved),
                    "peak_allocated_mib": mib(e2e_peak_allocated),
                    "peak_reserved_mib": mib(e2e_peak_reserved),
                    "incremental_peak_allocated_mib": mib(
                        e2e_peak_allocated - e2e_base_allocated
                    ),
                },
            }
            result["scales"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del batch

    result["q64_policy"] = "plain G-PCC passthrough; restoration network is not invoked"
    result["aggregate_over_enabled_scales"] = {
        "core_p50_ms_mean": statistics.mean(
            row["core_latency"]["p50_ms"] for row in result["scales"]
        ),
        "end_to_end_p50_ms_mean": statistics.mean(
            row["end_to_end_latency"]["p50_ms"] for row in result["scales"]
        ),
        "max_end_to_end_peak_allocated_mib": max(
            row["end_to_end_memory"]["peak_allocated_mib"] for row in result["scales"]
        ),
        "max_end_to_end_incremental_peak_allocated_mib": max(
            row["end_to_end_memory"]["incremental_peak_allocated_mib"]
            for row in result["scales"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
