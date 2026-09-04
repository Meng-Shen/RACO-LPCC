#!/usr/bin/env python3
"""One-frame OpenPCDet inference complexity benchmark (batch size 1)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPCDET_ROOT = REPO_ROOT / "OpenPCDet"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OPENPCDET_ROOT))

from integrations.openpcdet import install_openpcdet_compat

install_openpcdet_compat()

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu


MIB = 1024.0 ** 2


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max())}


def logger():
    log = logging.getLogger("pcdet-complexity")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    return log


def collect_tensors(value, out):
    if torch.is_tensor(value):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_tensors(item, out)


def indice_tensors(data):
    names = ("pair_fwd", "pair_bwd", "pair_mask_fwd_splits", "pair_mask_bwd_splits",
             "mask_argsort_fwd_splits", "mask_argsort_bwd_splits", "masks")
    out = []
    for name in names:
        if hasattr(data, name):
            try:
                collect_tensors(getattr(data, name), out)
            except Exception:
                pass
    return out


def unique_bytes(tensors):
    seen, total = set(), 0
    for tensor in tensors:
        key = (int(tensor.data_ptr()), int(tensor.numel()), int(tensor.element_size()))
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return int(total)


def profile_learned_macs(model, base_batch):
    sparse, dense, hooks = [], [], []
    sparse_classes = {"SubMConv3d", "SparseConv3d", "SparseInverseConv3d"}
    for name, module in model.named_modules():
        if module.__class__.__name__ in sparse_classes:
            def make_sparse(layer_name, layer_module):
                def hook(_module, inputs, output):
                    source = inputs[0]
                    key = getattr(layer_module, "indice_key", None)
                    data = output.indice_dict.get(key) if key is not None and hasattr(output, "indice_dict") else None
                    valid = int(output.features.shape[0])
                    cache_bytes = 0
                    if data is not None:
                        tensors = indice_tensors(data)
                        cache_bytes = unique_bytes(tensors)
                        pair = getattr(data, "pair_fwd", None)
                        if torch.is_tensor(pair):
                            valid = int((pair >= 0).sum().item())
                    cin, cout = int(layer_module.in_channels), int(layer_module.out_channels)
                    sparse.append({"name": layer_name, "class": layer_module.__class__.__name__,
                                   "input_active": int(source.features.shape[0]),
                                   "output_active": int(output.features.shape[0]),
                                   "effective_mappings": valid, "input_channels": cin,
                                   "output_channels": cout, "macs": int(valid * cin * cout),
                                   "indice_tensor_bytes": cache_bytes})
                return hook
            hooks.append(module.register_forward_hook(make_sparse(name, module)))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                 nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            def make_conv(layer_name, layer_module):
                def hook(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(tensor):
                        return
                    kernel = int(np.prod(layer_module.kernel_size))
                    per_output = int(layer_module.in_channels // layer_module.groups) * kernel
                    dense.append({"name": layer_name, "class": layer_module.__class__.__name__,
                                  "output_elements": int(tensor.numel()),
                                  "macs": int(tensor.numel() * per_output)})
                return hook
            hooks.append(module.register_forward_hook(make_conv(name, module)))
        elif isinstance(module, nn.Linear):
            def make_linear(layer_name, layer_module):
                def hook(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if torch.is_tensor(tensor):
                        vectors = int(tensor.numel() // layer_module.out_features)
                        dense.append({"name": layer_name, "class": "Linear", "vectors": vectors,
                                      "macs": int(vectors * layer_module.in_features * layer_module.out_features)})
                return hook
            hooks.append(module.register_forward_hook(make_linear(name, module)))
    with torch.inference_mode():
        model(dict(base_batch))
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    sparse_macs = int(sum(row["macs"] for row in sparse))
    dense_macs = int(sum(row["macs"] for row in dense))
    return {"sparse_layers": sparse, "dense_layers": dense,
            "sparse_macs": sparse_macs, "dense_macs": dense_macs,
            "total_learned_macs": sparse_macs + dense_macs,
            "flops_2_per_mac": 2 * (sparse_macs + dense_macs),
            "scope": "spconv effective mappings plus Conv/Linear learned MACs; voxelization, point grouping/search, NMS, BN and activations excluded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    cfg_from_yaml_file(str(args.config), cfg)
    dataset, _loader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger(), training=False,
    )
    cpu_item = dataset[0]
    cpu_batch = dataset.collate_batch([cpu_item])
    point_count = int(cpu_batch["points"].shape[0])
    frame_value = cpu_batch.get("frame_id", ["0"])
    frame_id = str(frame_value[0] if hasattr(frame_value, "__len__") else frame_value)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.checkpoint), logger=logger(), to_cpu=True)
    model.cuda().eval()
    parameters = int(sum(p.numel() for p in model.parameters()))
    baseline_alloc = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())

    base_gpu = dataset.collate_batch([dataset[0]])
    load_data_to_gpu(base_gpu)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(dict(base_gpu))
    torch.cuda.synchronize()
    mac_profile = profile_learned_macs(model, base_gpu)

    core_times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = time.perf_counter()
            model(dict(base_gpu))
            torch.cuda.synchronize()
            core_times.append((time.perf_counter() - start) * 1000.0)
    core_memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                   "incremental_peak_allocated_mib": (torch.cuda.max_memory_allocated() - baseline_alloc) / MIB}

    cpu_times, transfer_times, e2e_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            total_start = time.perf_counter()
            current = dataset.collate_batch([dataset[0]])
            cpu_done = time.perf_counter()
            load_data_to_gpu(current)
            torch.cuda.synchronize()
            transfer_done = time.perf_counter()
            model(current)
            torch.cuda.synchronize()
            done = time.perf_counter()
            cpu_times.append((cpu_done - total_start) * 1000.0)
            transfer_times.append((transfer_done - cpu_done) * 1000.0)
            e2e_times.append((done - total_start) * 1000.0)
    e2e_memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                  "incremental_peak_allocated_mib": (torch.cuda.max_memory_allocated() - baseline_alloc) / MIB}

    result = {
        "component": args.name, "dataset": "KITTI detection validation", "sample_id": frame_id,
        "input": {"points_after_dataset_filters": point_count,
                  "active_voxels": int(cpu_batch.get("voxels", np.empty((0,))).shape[0])},
        "model": {"parameters": parameters,
                  "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                  "fp32_weight_mib": parameters * 4 / MIB,
                  "fp16_bf16_weight_mib": parameters * 2 / MIB,
                  "sparse_conv_layers": len(mac_profile["sparse_layers"]),
                  "conv_linear_layers_profiled": len(mac_profile["dense_layers"]),
                  "checkpoint_bytes": args.checkpoint.stat().st_size},
        "compute": mac_profile,
        "latency": {"core_model_predict": stats(core_times),
                    "cpu_dataset_pipeline_including_cached_file_read": stats(cpu_times),
                    "host_to_device": stats(transfer_times),
                    "end_to_end_framework_predict": stats(e2e_times)},
        "memory": {"baseline_allocated_mib": baseline_alloc / MIB,
                   "baseline_reserved_mib": baseline_reserved / MIB,
                   "core": core_memory, "end_to_end": e2e_memory},
        "protocol": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                     "batch_size": 1, "warmup": args.warmup, "repeats": args.repeats,
                     "config": str(args.config), "checkpoint": str(args.checkpoint),
                     "core_input": "framework-preprocessed CUDA batch",
                     "end_to_end_note": "includes OpenPCDet dataset pipeline and cached local file read, H2D, network and prediction postprocess"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "params": parameters,
                      "core_p50_ms": result["latency"]["core_model_predict"]["p50_ms"],
                      "e2e_p50_ms": result["latency"]["end_to_end_framework_predict"]["p50_ms"]}, indent=2))


if __name__ == "__main__":
    main()
