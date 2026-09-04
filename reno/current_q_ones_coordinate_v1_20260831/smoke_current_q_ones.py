#!/usr/bin/env python3
"""Six-scale forward/backward smoke test for the all-one sparse backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchsparse.nn import functional as sparse_functional

from coordinate_residual import (
    CURRENT_SCALE_ONES_ARCHITECTURE,
    N_BY_Q,
    N_MAX,
    Q_STEPS_MM,
    CoordinateResidualNet,
    assert_anchor_alignment,
    build_residual_batch,
    decoded_point_clouds,
)


def configure_torchsparse() -> None:
    config = sparse_functional.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    sparse_functional.conv_config.set_global_conv_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_torchsparse()
    device = torch.device("cuda", 0)
    values = np.fromfile(args.point, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid Nx4 point cloud: {args.point}")
    xyz = np.ascontiguousarray(values.reshape(-1, 4)[:, :3])

    model = CoordinateResidualNet().to(device)
    source_architecture, loaded_tensors = model.load_coordinate_checkpoint(
        args.init_checkpoint, map_location="cpu"
    )
    model.train()
    model.zero_grad(set_to_none=True)

    forbidden = [
        name
        for name, _ in model.named_parameters()
        if any(token in name for token in ("prior", "occupancy", "fcg", "target_embedding"))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden parent/occupancy parameters remain: {forbidden}")

    rows = []
    total = torch.zeros((), device=device)
    for q_step_mm in Q_STEPS_MM:
        n = N_BY_Q[q_step_mm]
        batch = build_residual_batch([xyz], q_step_mm, n, device)
        if not torch.all(batch.input_features == 1):
            raise RuntimeError("input features are not all one")
        pred_all, output_coords = model(
            batch.input_coords, batch.input_features, N_MAX, q_step_mm
        )
        assert_anchor_alignment(output_coords, batch.anchor_coords)
        if pred_all.shape != (len(batch.anchor_coords), N_MAX, 3):
            raise RuntimeError(f"unexpected prediction shape at q={q_step_mm}: {pred_all.shape}")
        if not torch.all((pred_all >= 0) & (pred_all <= 1)):
            raise RuntimeError(f"normalized offsets out of range at q={q_step_mm}")
        decoded = decoded_point_clouds(
            pred_all[:, :n], output_coords, batch.origins_mm, q_step_mm
        )[0]
        if decoded.shape != (len(batch.anchor_coords) * n, 3):
            raise RuntimeError(f"unexpected decoded shape at q={q_step_mm}: {decoded.shape}")
        scale_loss = decoded.square().mean() * 1e-4 + pred_all.mean()
        total = total + scale_loss
        rows.append(
            {
                "q_step_mm": q_step_mm,
                "active_voxels": int(len(batch.input_coords)),
                "input_channels": int(batch.input_features.shape[1]),
                "input_feature_min": float(batch.input_features.min()),
                "input_feature_max": float(batch.input_features.max()),
                "n": n,
                "decoded_points": int(len(decoded)),
                "prediction_shape": list(pred_all.shape),
            }
        )

    total.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()]
    parameters_with_grad = sum(
        parameter.numel() for _, parameter in trainable if parameter.grad is not None
    )
    nonfinite_gradients = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise FloatingPointError(f"non-finite gradients: {nonfinite_gradients}")
    if parameters_with_grad == 0:
        raise RuntimeError("no model parameter received a gradient")

    total_parameters = sum(parameter.numel() for _, parameter in trainable)
    payload = {
        "status": "complete",
        "architecture": CURRENT_SCALE_ONES_ARCHITECTURE,
        "source_checkpoint_architecture": source_architecture,
        "initialized_compatible_tensors": loaded_tensors,
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "parameters_with_gradient": parameters_with_grad,
        "forbidden_parent_or_occupancy_parameters": forbidden,
        "scales": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("SMOKE_COMPLETE " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
