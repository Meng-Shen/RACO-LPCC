#!/usr/bin/env python3
"""Isolated one-frame benchmarks for the RENO encoder and LRProxy router.

Run each component in a fresh process so CUDA allocator state from one model
cannot contaminate another model's peak-memory measurement.

Use ``measure_semantickitti_minkunet_complexity.py`` for MinkUNet measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


MIB = 1024.0 ** 2


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def parameter_summary(model):
    parameters = list(model.parameters())
    modules = list(model.modules())
    return {
        "total_parameters": int(sum(p.numel() for p in parameters)),
        "trainable_parameters": int(sum(p.numel() for p in parameters if p.requires_grad)),
        "parameter_bytes_current_dtype": int(sum(p.numel() * p.element_size() for p in parameters)),
        "parameter_mib_current_dtype": float(sum(p.numel() * p.element_size() for p in parameters) / MIB),
        "buffer_bytes_current_dtype": int(sum(b.numel() * b.element_size() for b in model.buffers())),
        "module_count": int(len(modules)),
        "linear_layer_count": int(sum(m.__class__.__name__ == "Linear" for m in modules)),
        "sparse_conv_like_layer_count": int(sum(
            (
                m.__class__.__name__ in {
                    "Conv3d", "SubMConv3d", "SparseConv3d",
                    "MinkowskiConvolution", "MinkowskiConvolutionTranspose",
                }
            )
            for m in modules
        )),
    }


def device_summary(device):
    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update({
            "gpu_name": props.name,
            "gpu_total_memory_mib": float(props.total_memory / MIB),
        })
    return result


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_baseline(device):
    if device.type != "cuda":
        return {"allocated_mib": 0.0, "reserved_mib": 0.0}
    cuda_sync(device)
    return {
        "allocated_mib": float(torch.cuda.memory_allocated(device) / MIB),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / MIB),
    }


def reset_peak(device):
    if device.type == "cuda":
        cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_summary(device, baseline_allocated_mib):
    if device.type != "cuda":
        return {
            "peak_allocated_mib": 0.0,
            "peak_reserved_mib": 0.0,
            "incremental_peak_allocated_mib": 0.0,
        }
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / MIB)
    return {
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / MIB),
        "incremental_peak_allocated_mib": max(0.0, peak_allocated - baseline_allocated_mib),
    }


def read_points(path):
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Invalid KITTI point cloud: {path}")
    return np.ascontiguousarray(raw.reshape(-1, 4))


def benchmark_reno(args, device, points):
    repo_root = args.repo_root.resolve()
    reno_dir = repo_root / "reno"
    sys.path[:0] = [str(reno_dir), str(repo_root)]
    from reno_rates import configure_torchsparse, encode_tensor, load_model, points_to_sparse

    configure_torchsparse()
    load_start = time.perf_counter()
    model = load_model(args.reno_root, args.reno_checkpoint, 32, 3, device)
    cuda_sync(device)
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    model.eval()
    model_info = parameter_summary(model)
    model_info["checkpoint"] = str(args.reno_checkpoint.resolve())
    model_info["checkpoint_bytes"] = args.reno_checkpoint.stat().st_size
    model_info["initialization_and_checkpoint_load_ms"] = load_time_ms
    baseline = cuda_baseline(device)
    xyz = points[:, :3]
    rates = []

    def encode_once(posq):
        reset_peak(device)
        start = time.perf_counter()
        sparse, offset = points_to_sparse(xyz, posq, device)
        cuda_sync(device)
        after_preprocess = time.perf_counter()
        with torch.inference_mode():
            base_coords, base_feats, payload = encode_tensor(model, sparse)
        cuda_sync(device)
        done = time.perf_counter()
        # Exact byte count of reno_rates.write_bitstream without filesystem I/O.
        container_bytes = (
            4 + 12 + 4 + int(base_coords.nbytes) + int(base_feats.nbytes) + len(payload)
        )
        memory = peak_summary(device, baseline["allocated_mib"])
        del sparse, base_coords, base_feats, payload
        return {
            "preprocess_ms": (after_preprocess - start) * 1000.0,
            "encode_ms": (done - after_preprocess) * 1000.0,
            "end_to_end_ms": (done - start) * 1000.0,
            "container_bytes": int(container_bytes),
            "memory": memory,
            "offset_mm": [int(v) for v in offset],
        }

    for rate_id, posq in enumerate(args.quant_steps_mm):
        for _ in range(args.reno_warmup):
            encode_once(float(posq))
        samples = [encode_once(float(posq)) for _ in range(args.reno_repeats)]
        coords_mm = np.rint(xyz.astype(np.float64) * 1000.0).astype(np.int64)
        origin = coords_mm.min(axis=0)
        quantized = np.rint((coords_mm - origin) / float(posq)).astype(np.int32)
        byte_counts = [item["container_bytes"] for item in samples]
        memory_keys = samples[0]["memory"].keys()
        rates.append({
            "rate_id_coarse_to_fine": int(rate_id),
            "quant_step_mm": int(posq),
            "scale": f"1/{int(posq)}",
            "unique_quantized_points": int(np.unique(quantized, axis=0).shape[0]),
            "preprocess": percentile_summary([item["preprocess_ms"] for item in samples]),
            "encode_without_disk_io": percentile_summary([item["encode_ms"] for item in samples]),
            "end_to_end_without_disk_io": percentile_summary([item["end_to_end_ms"] for item in samples]),
            "bitstream_bytes": int(round(float(np.median(byte_counts)))),
            "bits_per_original_point": float(np.median(byte_counts) * 8.0 / len(points)),
            "memory_max_over_repeats": {
                key: float(max(item["memory"][key] for item in samples)) for key in memory_keys
            },
        })
        print(f"[RENO] q={posq} mm complete", flush=True)

    return {
        "component": "reno_encoder",
        "model": model_info,
        "cuda_baseline_after_model_load": baseline,
        "rates_coarse_to_fine": rates,
        "timing_scope": "quantization+sparse construction+neural entropy coding+arithmetic coding+in-memory container size; disk I/O excluded",
    }


def benchmark_router(args, device, points):
    repo_root = args.repo_root.resolve()
    tools_dir = repo_root / "OpenPCDet" / "tools"
    router_dir = repo_root / "routing" / "lrproxy"
    sys.path[:0] = [str(router_dir), str(tools_dir)]
    from cpu_voxelizer import voxelize_points
    from gpu_voxelizer import voxelize_points_gpu
    from lrproxy import (
        LRProxy,
        count_parameters as router_parameter_breakdown,
        select_xyz_features,
    )
    voxel_size = np.asarray(args.router_voxel_size, dtype=np.float32)
    pc_range = np.asarray(args.router_point_cloud_range, dtype=np.float32)
    max_voxels = int(args.router_max_voxels)
    load_start = time.perf_counter()
    checkpoint = torch.load(args.router_checkpoint, map_location="cpu")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["model"].items()
    }
    loss_scales = state["loss_scales"].float().tolist()
    mean_log_bpp = torch.cumsum(
        state["mean_log_increments"].float(), dim=0
    ).tolist()
    model = LRProxy(
        feat_dim=256,
        loss_scales=loss_scales,
        mean_log_bpp=mean_log_bpp,
    )
    load_report = model.load_full_checkpoint(args.router_checkpoint)
    model.to(device).eval()
    cuda_sync(device)
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    model_info = parameter_summary(model)
    model_info["checkpoint"] = str(args.router_checkpoint.resolve())
    model_info["checkpoint_bytes"] = args.router_checkpoint.stat().st_size
    model_info["initialization_and_checkpoint_load_ms"] = load_time_ms
    model_info["load_report"] = load_report
    model_info["parameter_breakdown"] = router_parameter_breakdown(model)
    model_info["mac_formula_excluding_bn_activations_pooling"] = (
        "10336 * active_voxels + 527360"
    )
    baseline = cuda_baseline(device)

    def prepare_cpu():
        return voxelize_points(
            points,
            voxel_size=voxel_size,
            pc_range=pc_range,
            max_voxels=max_voxels,
            use_abs_xyz=True,
            include_intensity=False,
        )

    with torch.inference_mode():
        for _ in range(args.router_warmup):
            if args.router_preprocessor == "gpu":
                warm_points = torch.from_numpy(points).to(device)
                warm_vf_t, warm_vc = voxelize_points_gpu(
                    warm_points,
                    voxel_size,
                    pc_range,
                    max_voxels=max_voxels,
                    use_abs_xyz=True,
                    include_intensity=False,
                )
                warm_vf_t = select_xyz_features(warm_vf_t)[None, ...]
            else:
                warm_vf, warm_vc = prepare_cpu()
                warm_vf_t = torch.from_numpy(warm_vf[:, 4:7]).to(device)[None, ...]
            model(warm_vf_t)
            cuda_sync(device)
            del warm_vf_t
            if args.router_preprocessor == "gpu":
                del warm_points, warm_vc

    samples = []
    final_output = None
    for _ in range(args.router_repeats):
        reset_peak(device)
        start = time.perf_counter()
        if args.router_preprocessor == "gpu":
            points_t = torch.from_numpy(points).to(device, non_blocking=False)
            cuda_sync(device)
            transfer_done = time.perf_counter()
            features_t, voxel_coords = voxelize_points_gpu(
                points_t,
                voxel_size,
                pc_range,
                max_voxels=max_voxels,
                use_abs_xyz=True,
                include_intensity=False,
            )
            features_t = select_xyz_features(features_t)[None, ...]
            cuda_sync(device)
            voxel_done = time.perf_counter()
        else:
            voxel_features, voxel_coords = prepare_cpu()
            voxel_done = time.perf_counter()
            features_t = torch.from_numpy(voxel_features[:, 4:7]).to(
                device, non_blocking=False
            )[None, ...]
            cuda_sync(device)
            transfer_done = time.perf_counter()
        with torch.inference_mode():
            output = model(features_t)
        cuda_sync(device)
        forward_done = time.perf_counter()
        final_output = {
            "loss_pred": output["loss_pred"].detach().cpu().tolist()[0],
            "bpp_pred": output["bpp_pred"].detach().cpu().tolist()[0],
        }
        samples.append({
            "voxelization_ms": (
                (voxel_done - transfer_done) * 1000.0
                if args.router_preprocessor == "gpu"
                else (voxel_done - start) * 1000.0
            ),
            "host_to_device_ms": (
                (transfer_done - start) * 1000.0
                if args.router_preprocessor == "gpu"
                else (transfer_done - voxel_done) * 1000.0
            ),
            "network_forward_ms": (
                (forward_done - voxel_done) * 1000.0
                if args.router_preprocessor == "gpu"
                else (forward_done - transfer_done) * 1000.0
            ),
            "end_to_end_ms": (forward_done - start) * 1000.0,
            "active_voxels": int(features_t.shape[1]),
            "memory": peak_summary(device, baseline["allocated_mib"]),
        })
        del features_t, output
        if args.router_preprocessor == "gpu":
            del points_t, voxel_coords

    memory_keys = samples[0]["memory"].keys()
    return {
        "component": "lrproxy_absolute_loss_monotonic_rate_router",
        "model": model_info,
        "cuda_baseline_after_model_load": baseline,
        "input_configuration": {
            "geometry_only": True,
            "input_feature_channels": 3,
            "voxel_size_m": voxel_size.tolist(),
            "point_cloud_range_m": pc_range.tolist(),
            "max_voxels": max_voxels,
            "preprocessor": args.router_preprocessor,
            "router_variant": "lrproxy",
            "active_voxels": int(round(np.median([s["active_voxels"] for s in samples]))),
        },
        f"{args.router_preprocessor}_voxelization": percentile_summary(
            [s["voxelization_ms"] for s in samples]
        ),
        "host_to_device": percentile_summary([s["host_to_device_ms"] for s in samples]),
        "network_forward": percentile_summary([s["network_forward_ms"] for s in samples]),
        "end_to_end": percentile_summary([s["end_to_end_ms"] for s in samples]),
        "memory_max_over_repeats": {
            key: float(max(s["memory"][key] for s in samples)) for key in memory_keys
        },
        "smoke_output": final_output,
        "checkpoint_outputs_loaded": True,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("reno", "router"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--frame-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reno-root", type=Path, default=Path("/public/DATA/sm/RENO"))
    parser.add_argument("--reno-checkpoint", type=Path)
    parser.add_argument("--quant-steps-mm", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--reno-warmup", type=int, default=1)
    parser.add_argument("--reno-repeats", type=int, default=7)
    parser.add_argument(
        "--router-checkpoint", "--router-legacy-checkpoint",
        dest="router_checkpoint", type=Path,
    )
    parser.add_argument(
        "--router-preprocessor", choices=("gpu", "cpu"), default="gpu"
    )
    parser.add_argument(
        "--router-voxel-size", type=float, nargs=3, default=[0.16, 0.16, 0.16]
    )
    parser.add_argument(
        "--router-point-cloud-range",
        type=float,
        nargs=6,
        default=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    )
    parser.add_argument("--router-max-voxels", type=int, default=50000)
    parser.add_argument("--router-warmup", type=int, default=5)
    parser.add_argument("--router-repeats", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    points = read_points(args.frame_bin)
    common = {
        "benchmark_version": "routing_components_one_frame_v2",
        "component_process_pid": os.getpid(),
        "frame": str(args.frame_bin.resolve()),
        "frame_id": args.frame_bin.stem,
        "original_points": int(len(points)),
        "input_bytes": int(args.frame_bin.stat().st_size),
        "device_environment": device_summary(device),
    }
    if args.component == "reno":
        if args.reno_checkpoint is None:
            raise ValueError("--reno-checkpoint is required")
        result = benchmark_reno(args, device, points)
    else:
        if args.router_checkpoint is None:
            raise ValueError("--router-checkpoint is required")
        result = benchmark_router(args, device, points)
    common.update(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(common, indent=2), encoding="utf-8")
    print(json.dumps(common, indent=2), flush=True)


if __name__ == "__main__":
    main()
