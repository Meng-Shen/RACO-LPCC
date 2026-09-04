#!/usr/bin/env python3
"""Restore G-PCC geometry with the current-q all-one sparse model.

The network consumes q-level decoded coordinates directly, with one constant
feature per active voxel.  It uses no parent-scale tensor, occupancy code,
original high-resolution coordinate, or additional side information.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchsparse.nn import functional as sparse_functional


Q_STEPS_MM = (2048, 1024, 512, 256, 128, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root", "--reno-root", dest="model_root", type=Path, required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--points-dir", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--rate-ids", default="0,1,2,3,4,5")
    parser.add_argument(
        "--passthrough-rate-ids",
        default="",
        help="Comma-separated rates written as plain G-PCC geometry without restoration.",
    )
    return parser.parse_args()


def configure_torchsparse() -> None:
    config = sparse_functional.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    sparse_functional.conv_config.set_global_conv_config(config)


def read_ids(path: Path, max_frames: int) -> list[str]:
    frame_ids = [line.strip().zfill(6) for line in path.read_text().splitlines() if line.strip()]
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError(f"no frame ids found in {path}")
    return frame_ids


def load_xyz(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"expected non-empty Nx4 float32 points: {path}")
    return np.ascontiguousarray(values.reshape(-1, 4)[:, :3])


def simulate_gpcc_decoded_xyz(points_xyz_m: np.ndarray, q_step_mm: int) -> np.ndarray:
    """Match the geometry used by the existing fixed G-PCC AP baseline."""

    coords_mm = np.rint(points_xyz_m.astype(np.float64) * 1000.0).astype(np.int64)
    origin_mm = coords_mm.min(axis=0)
    anchors = np.rint(
        (coords_mm - origin_mm[None, :]).astype(np.float64) / float(q_step_mm)
    ).astype(np.int64)
    anchors = np.unique(anchors, axis=0)
    decoded_mm = origin_mm[None, :] + anchors * int(q_step_mm)
    return np.ascontiguousarray(decoded_mm.astype(np.float32) * 0.001)


def load_model(checkpoint: Path, device: torch.device):
    from coordinate_residual import (
        N_MAX,
        CURRENT_SCALE_ONES_ARCHITECTURE,
        CoordinateResidualNet,
    )

    state = torch.load(checkpoint, map_location="cpu")
    if state.get("architecture") != CURRENT_SCALE_ONES_ARCHITECTURE:
        raise ValueError(f"unexpected coordinate architecture: {state.get('architecture')!r}")
    model = CoordinateResidualNet(channels=32, kernel_size=3, n_max=N_MAX).to(device)
    model.load_coordinate_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    model_root = str(args.model_root.resolve())
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    from coordinate_residual import (
        N_BY_Q,
        N_MAX,
        assert_anchor_alignment,
        build_inference_batch_from_decoded_xyz,
        decoded_point_clouds,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    configure_torchsparse()

    requested_rates = [int(item) for item in args.rate_ids.replace(",", " ").split()]
    if not requested_rates or any(rate < 0 or rate >= len(Q_STEPS_MM) for rate in requested_rates):
        raise ValueError("--rate-ids must select from 0..5")
    passthrough_rates = {
        int(item)
        for item in args.passthrough_rate_ids.replace(",", " ").split()
    }
    if not passthrough_rates.issubset(requested_rates):
        raise ValueError("--passthrough-rate-ids must be a subset of --rate-ids")
    frame_ids = read_ids(args.split, args.max_frames)
    shard_ids = frame_ids[rank::world_size]
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    for rate_id in requested_rates:
        (args.decoded_dir / f"rate_{rate_id}").mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for frame_index, frame_id in enumerate(shard_ids, 1):
            raw_xyz = load_xyz(args.points_dir / f"{frame_id}.bin")
            for rate_id in requested_rates:
                q_step_mm = Q_STEPS_MM[rate_id]
                n = N_BY_Q[q_step_mm]
                gpcc_xyz = simulate_gpcc_decoded_xyz(raw_xyz, q_step_mm)
                if rate_id in passthrough_rates:
                    restored_xyz = torch.as_tensor(gpcc_xyz, dtype=torch.float32, device=device)
                    input_active_voxels = len(gpcc_xyz)
                else:
                    batch = build_inference_batch_from_decoded_xyz(
                        gpcc_xyz, q_step_mm, n, device
                    )
                    pred_all, anchor_coords = model(
                        batch.input_coords, batch.input_features, N_MAX, q_step_mm
                    )
                    assert_anchor_alignment(anchor_coords, batch.anchor_coords)
                    restored_xyz = decoded_point_clouds(
                        pred_all[:, :n], anchor_coords, batch.origins_mm, q_step_mm
                    )[0]
                    input_active_voxels = int(batch.input_coords.shape[0])
                restored = torch.zeros(
                    (restored_xyz.shape[0], 4), dtype=torch.float32, device=device
                )
                restored[:, :3] = restored_xyz
                output = args.decoded_dir / f"rate_{rate_id}" / f"{frame_id}.bin"
                restored.cpu().numpy().tofile(output)
                rows.append(
                    {
                        "frame_id": frame_id,
                        "rate_id": rate_id,
                        "q_step_mm": q_step_mm,
                        "mode": "gpcc_passthrough" if rate_id in passthrough_rates else "restored",
                        "original_points": len(raw_xyz),
                        "gpcc_decoded_points": len(gpcc_xyz),
                        "input_active_voxels": input_active_voxels,
                        "restored_points": int(restored.shape[0]),
                        "output_bytes": output.stat().st_size,
                    }
                )
            if frame_index == 1 or frame_index % 100 == 0 or frame_index == len(shard_ids):
                print(
                    f"rank={rank}/{world_size} frames={frame_index}/{len(shard_ids)} "
                    f"last={frame_id} elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    manifest = args.summary_dir / f"restore_manifest_rank_{rank:02d}.csv"
    with manifest.open("w", newline="") as handle:
        fields = list(rows[0]) if rows else ["frame_id", "rate_id", "q_step_mm"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"COMPLETE rank={rank} rows={len(rows)} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
