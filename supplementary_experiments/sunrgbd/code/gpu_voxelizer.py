#!/usr/bin/env python3
"""CUDA voxelization for the geometry-only sparse routing proxy.

The feature definition matches ``train_cost_proxy.voxelize_points``:
relative xyz (3), log-density (1), and normalized absolute xyz (3).
Unlike the legacy NumPy implementation, coordinates never make a CPU
round-trip after the raw frame is copied to CUDA.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch


def _as_device_tensor(
    value: Union[torch.Tensor, Iterable[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(list(value), device=device, dtype=dtype)


@torch.no_grad()
def voxelize_points_gpu(
    points: torch.Tensor,
    voxel_size: Union[torch.Tensor, Sequence[float]],
    pc_range: Union[torch.Tensor, Sequence[float]],
    max_voxels: int,
    use_abs_xyz: bool = True,
    include_intensity: bool = False,
    random_subsample: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voxelize one point cloud entirely on its current CUDA device.

    Args:
        points: ``[N, >=3]`` float tensor already on CUDA.
        random_subsample: randomly select active voxels only when their count
            exceeds ``max_voxels``.  Inference should leave this disabled for
            deterministic output; training can enable it to match the legacy
            augmentation intent.

    Returns:
        ``voxel_features`` and int32 ``voxel_coords_zyx``, both on CUDA.
    """
    if not torch.is_tensor(points) or points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be a [N, >=3] tensor")
    if points.device.type != "cuda":
        raise ValueError("voxelize_points_gpu expects points already on CUDA")
    if include_intensity and points.shape[1] < 4:
        raise ValueError("include_intensity=True requires a fourth point channel")

    device = points.device
    points = points.float().contiguous()
    voxel_size_t = _as_device_tensor(
        voxel_size, device=device, dtype=torch.float32
    ).reshape(3)
    pc_range_t = _as_device_tensor(
        pc_range, device=device, dtype=torch.float32
    ).reshape(6)
    if bool(torch.any(voxel_size_t <= 0)):
        raise ValueError("voxel_size entries must be positive")

    feature_dim = (7 if use_abs_xyz else 4) + int(include_intensity)
    xyz = points[:, :3]
    mask = torch.all(xyz >= pc_range_t[:3], dim=1) & torch.all(
        xyz < pc_range_t[3:], dim=1
    )
    points = points[mask]
    if points.shape[0] == 0:
        return (
            torch.zeros((1, feature_dim), device=device, dtype=torch.float32),
            torch.zeros((1, 3), device=device, dtype=torch.int32),
        )

    coords_xyz = torch.floor(
        (points[:, :3] - pc_range_t[:3]) / voxel_size_t
    ).to(torch.int64)
    grid_size = torch.floor(
        (pc_range_t[3:] - pc_range_t[:3]) / voxel_size_t
    ).to(torch.int64)
    valid = torch.all(coords_xyz >= 0, dim=1) & torch.all(
        coords_xyz < grid_size, dim=1
    )
    coords_xyz = coords_xyz[valid]
    points = points[valid]
    if points.shape[0] == 0:
        return (
            torch.zeros((1, feature_dim), device=device, dtype=torch.float32),
            torch.zeros((1, 3), device=device, dtype=torch.int32),
        )

    # A sorted scalar key reproduces NumPy's lexicographic x,y,z ordering but
    # avoids the slower torch.unique(..., dim=0) path.
    keys = (
        (coords_xyz[:, 0] * grid_size[1] + coords_xyz[:, 1])
        * grid_size[2]
        + coords_xyz[:, 2]
    )
    unique_keys, inverse, counts = torch.unique(
        keys, sorted=True, return_inverse=True, return_counts=True
    )

    if max_voxels > 0 and unique_keys.shape[0] > int(max_voxels):
        if random_subsample:
            keep = torch.randperm(
                unique_keys.shape[0], device=device, generator=generator
            )[: int(max_voxels)]
            keep = torch.sort(keep).values
        else:
            # Deterministic inference fallback, ordered lexicographically.
            keep = torch.arange(int(max_voxels), device=device)
        old_to_new = torch.full(
            (unique_keys.shape[0],), -1, device=device, dtype=torch.int64
        )
        old_to_new[keep] = torch.arange(keep.shape[0], device=device)
        point_keep = old_to_new[inverse] >= 0
        inverse = old_to_new[inverse[point_keep]]
        points = points[point_keep]
        unique_keys = unique_keys[keep]
        counts = torch.bincount(inverse, minlength=unique_keys.shape[0])

    yz_stride = grid_size[1] * grid_size[2]
    unique_x = torch.div(unique_keys, yz_stride, rounding_mode="floor")
    remainder = unique_keys - unique_x * yz_stride
    unique_y = torch.div(remainder, grid_size[2], rounding_mode="floor")
    unique_z = remainder - unique_y * grid_size[2]
    unique_coords_xyz = torch.stack((unique_x, unique_y, unique_z), dim=1)

    voxel_count = unique_keys.shape[0]
    sum_xyz = torch.zeros(
        (voxel_count, 3), device=device, dtype=torch.float32
    )
    sum_xyz.index_add_(0, inverse, points[:, :3])
    counts_f = counts.to(torch.float32).reshape(-1, 1)
    mean_xyz = sum_xyz / counts_f.clamp_min(1.0)

    mean_intensity = None
    if include_intensity:
        sum_intensity = torch.zeros(
            (voxel_count, 1), device=device, dtype=torch.float32
        )
        sum_intensity.index_add_(0, inverse, points[:, 3:4])
        mean_intensity = sum_intensity / counts_f.clamp_min(1.0)

    voxel_centers = (
        pc_range_t[:3]
        + (unique_coords_xyz.to(torch.float32) + 0.5) * voxel_size_t
    )
    rel_xyz = (mean_xyz - voxel_centers) / voxel_size_t
    density = torch.log1p(counts_f) / torch.log(
        torch.tensor(64.0, device=device, dtype=torch.float32)
    )
    density = density.clamp_(0.0, 1.0)

    parts = [rel_xyz]
    if mean_intensity is not None:
        parts.append(mean_intensity)
    parts.append(density)
    if use_abs_xyz:
        abs_xyz_norm = (
            (mean_xyz - pc_range_t[:3])
            / (pc_range_t[3:] - pc_range_t[:3] + 1e-6)
        )
        parts.append(abs_xyz_norm * 2.0 - 1.0)
    voxel_features = torch.cat(parts, dim=1).contiguous()
    voxel_coords_zyx = unique_coords_xyz[:, [2, 1, 0]].to(torch.int32).contiguous()
    return voxel_features, voxel_coords_zyx


@torch.no_grad()
def voxelize_batch_gpu(
    point_clouds: Sequence[torch.Tensor],
    voxel_size: Union[torch.Tensor, Sequence[float]],
    pc_range: Union[torch.Tensor, Sequence[float]],
    max_voxels: int,
    use_abs_xyz: bool = True,
    include_intensity: bool = False,
    random_subsample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voxelize a list of frames and prepend spconv batch coordinates."""
    all_features = []
    all_coords = []
    for batch_index, points in enumerate(point_clouds):
        features, coords = voxelize_points_gpu(
            points,
            voxel_size,
            pc_range,
            max_voxels,
            use_abs_xyz=use_abs_xyz,
            include_intensity=include_intensity,
            random_subsample=random_subsample,
        )
        batch_column = torch.full(
            (coords.shape[0], 1),
            batch_index,
            device=coords.device,
            dtype=torch.int32,
        )
        all_features.append(features)
        all_coords.append(torch.cat((batch_column, coords), dim=1))
    return torch.cat(all_features, dim=0), torch.cat(all_coords, dim=0)
