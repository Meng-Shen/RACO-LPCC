#!/usr/bin/env python3
"""Compare legacy preprocessing with decoder-ready GPU inference paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTORER_ROOT = PROJECT_ROOT / "reno" / "current_q_ones_coordinate_v1_20260831"
if str(RESTORER_ROOT) not in sys.path:
    sys.path.insert(0, str(RESTORER_ROOT))

from measure_kitti_geometry_restorer_complexity import configure_torchsparse, latency_stats
from coordinate_residual import (
    N_BY_Q,
    N_MAX,
    CoordinateResidualNet,
    build_decoder_target,
    build_inference_batch_from_anchors,
    build_inference_batch_from_decoded_xyz,
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
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def timed(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def sort_xyz(value: torch.Tensor):
    array = value.detach().cpu().numpy()
    order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
    return array[order]


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    configure_torchsparse()
    model = CoordinateResidualNet(channels=32, kernel_size=3, n_max=N_MAX).to(device)
    model.load_coordinate_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    raw_xyz = load_xyz(args.frame)
    rows = []

    with torch.inference_mode():
        for q_step_mm in Q_STEPS_MM:
            n = N_BY_Q[q_step_mm]
            # In production these are supplied by the G-PCC decoder.  The
            # simulation is deliberately outside all decoder-ready timings.
            gpcc_xyz = simulate_gpcc_decoded_xyz(raw_xyz, q_step_mm)
            decoder_target = build_decoder_target(gpcc_xyz, q_step_mm)
            anchors_cpu = decoder_target.anchor_coords
            origin_cpu = decoder_target.origin_mm
            decoded_gpu = torch.as_tensor(gpcc_xyz, device=device, dtype=torch.float32)
            anchors_gpu = torch.as_tensor(anchors_cpu, device=device, dtype=torch.int32)
            origin_gpu = torch.as_tensor(origin_cpu, device=device, dtype=torch.float32)

            def run_batch(batch):
                pred, coords = model(
                    batch.input_coords, batch.input_features, N_MAX, q_step_mm
                )
                return decoded_point_clouds(
                    pred[:, :n], coords, batch.origins_mm, q_step_mm
                )[0]

            def legacy_decoded_cpu():
                return run_batch(build_residual_batch([gpcc_xyz], q_step_mm, n, device))

            def fast_decoded_cpu():
                return run_batch(
                    build_inference_batch_from_decoded_xyz(
                        gpcc_xyz, q_step_mm, n, device, origin_cpu
                    )
                )

            def fast_anchors_cpu():
                return run_batch(
                    build_inference_batch_from_anchors(
                        anchors_cpu, origin_cpu, q_step_mm, n, device
                    )
                )

            def fast_decoded_gpu():
                return run_batch(
                    build_inference_batch_from_decoded_xyz(
                        decoded_gpu, q_step_mm, n, device, origin_gpu
                    )
                )

            def fast_anchors_gpu():
                return run_batch(
                    build_inference_batch_from_anchors(
                        anchors_gpu, origin_gpu, q_step_mm, n, device
                    )
                )

            methods = {
                "legacy_decoded_cpu": legacy_decoded_cpu,
                "fast_decoded_cpu": fast_decoded_cpu,
                "fast_anchors_cpu": fast_anchors_cpu,
                "fast_decoded_gpu": fast_decoded_gpu,
                "fast_anchors_gpu": fast_anchors_gpu,
            }
            for _ in range(args.warmup):
                for function in methods.values():
                    function()
            torch.cuda.synchronize()

            outputs = {name: function() for name, function in methods.items()}
            reference = sort_xyz(outputs["legacy_decoded_cpu"])
            max_abs_diff = {
                name: float(np.max(np.abs(sort_xyz(output) - reference)))
                for name, output in outputs.items()
            }
            timings = {}
            for name, function in methods.items():
                samples = []
                for _ in range(args.repeats):
                    _, elapsed_ms = timed(function)
                    samples.append(elapsed_ms)
                timings[name] = latency_stats(samples)

            row = {
                "q_step_mm": q_step_mm,
                "active_voxels": int(len(anchors_cpu)),
                "timings": timings,
                "max_abs_output_diff_m": max_abs_diff,
                "speedup_fast_anchors_cpu_vs_legacy": (
                    timings["legacy_decoded_cpu"]["p50_ms"]
                    / timings["fast_anchors_cpu"]["p50_ms"]
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    result = {
        "frame": str(args.frame),
        "input_points": int(len(raw_xyz)),
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "scope": "all timings exclude G-PCC encoding/decoding or raw-cloud quantization",
            "legacy": "decoded CPU XYZ -> CPU requantize/unique/sort -> GPU -> network -> decode",
            "fast_anchors_cpu": "decoder CPU integer anchors+origin -> GPU -> all-one sparse input -> network -> decode",
            "fast_anchors_gpu": "decoder integer anchors+origin already on GPU -> all-one sparse input -> network -> decode",
        },
        "scales": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
