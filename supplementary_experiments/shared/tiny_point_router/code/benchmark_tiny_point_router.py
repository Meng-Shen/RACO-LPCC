#!/usr/bin/env python3
"""Single-object latency/complexity smoke benchmark for TinyPoint router."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


MIB = 1024.0 ** 2


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_dir.resolve()))
    from tiny_point_absolute_loss_monotonic_rate_proxy import (
        TinyPointAbsoluteLossMonotonicRateProxy, count_parameters,
    )

    array = np.load(args.points, mmap_mode="r")
    xyz = np.ascontiguousarray(np.asarray(array[args.index])[:, :3], dtype=np.float32).copy()
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    model = TinyPointAbsoluteLossMonotonicRateProxy(
        feat_dim=256, loss_scales=[1.0] * 6,
        mean_log_bpp=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], input_channels=3,
    ).cuda().eval()
    breakdown = count_parameters(model)
    baseline = int(torch.cuda.memory_allocated())
    points = torch.from_numpy(xyz[None]).cuda()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(points)
    torch.cuda.synchronize()

    core = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = model(points)
            torch.cuda.synchronize()
            core.append((time.perf_counter() - start) * 1000.0)
    core_memory = (torch.cuda.max_memory_allocated() - baseline) / MIB

    e2e = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = time.perf_counter()
            current = torch.from_numpy(xyz.copy()[None]).cuda()
            result = model(current)
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - start) * 1000.0)
            del current, result
    e2e_memory = (torch.cuda.max_memory_allocated() - baseline) / MIB

    n = int(xyz.shape[0])
    point_mlp_macs = n * (3 * 32 + 32 * 64 + 64 * 128)
    global_mlp_macs = 256 * 256
    six_loss_head_macs = 6 * (256 * 256 + 256)
    rate_head_macs = 256 * 256 + 256 * 6
    total_macs = point_mlp_macs + global_mlp_macs + six_loss_head_macs + rate_head_macs
    result = {
        "component": "TinyPoint direct-six-loss monotonic-rate router",
        "dataset": args.dataset, "sample_index": args.index, "points": n,
        "parameters": breakdown,
        "weight_mib": {"fp32": breakdown["total"] * 4 / MIB, "fp16_bf16": breakdown["total"] * 2 / MIB},
        "compute": {"point_mlp_macs": point_mlp_macs, "global_mlp_macs": global_mlp_macs,
                    "six_loss_head_macs": six_loss_head_macs, "rate_head_macs": rate_head_macs,
                    "total_macs": total_macs, "flops_2_per_mac": 2 * total_macs},
        "latency": {"core": stats(core), "end_to_end_in_memory": stats(e2e)},
        "memory": {"core_incremental_peak_mib": core_memory, "e2e_incremental_peak_mib": e2e_memory},
        "output_contract": {"keys": sorted(output), "loss_shape": list(output["loss_pred"].shape),
                            "bpp_shape": list(output["bpp_pred"].shape),
                            "bpp_monotonic": bool(torch.all(torch.diff(output["bpp_pred"], dim=1) >= 0).item()),
                            "loss_heads_independent": True},
        "protocol": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                     "batch_size": 1, "warmup": args.warmup, "repeats": args.repeats,
                     "disk_io_excluded": True, "voxelization": False,
                     "sparse_coordinate_indexing": False, "knn_or_fps": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "params": breakdown,
                      "macs": total_macs, "core": result["latency"]["core"],
                      "e2e": result["latency"]["end_to_end_in_memory"],
                      "contract": result["output_contract"]}, indent=2))


if __name__ == "__main__":
    main()
