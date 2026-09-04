#!/usr/bin/env python3
"""NumPy voxelization used by current LRProxy router benchmarks.

This module is intentionally independent from the retired AP-drop/JUQP proxy
trainer.  Its feature layout matches :mod:`gpu_voxelizer`: relative XYZ,
optional mean intensity, log-density, and optional normalized absolute XYZ.
"""

from typing import Tuple

import numpy as np


def voxelize_points(
    points: np.ndarray,
    voxel_size: np.ndarray,
    pc_range: np.ndarray,
    max_voxels: int,
    use_abs_xyz: bool = True,
    include_intensity: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize one point cloud on CPU using mean feature aggregation."""
    feature_dim = (7 if use_abs_xyz else 4) + int(include_intensity)
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3])
        & (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4])
        & (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
    )
    points = points[mask]
    if points.shape[0] == 0:
        return (
            np.zeros((1, feature_dim), dtype=np.float32),
            np.zeros((1, 3), dtype=np.int32),
        )

    xyz = points[:, :3]
    coords_xyz = np.floor((xyz - pc_range[:3]) / voxel_size).astype(np.int32)
    grid_size = np.floor((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)
    valid = (
        (coords_xyz[:, 0] >= 0) & (coords_xyz[:, 0] < grid_size[0])
        & (coords_xyz[:, 1] >= 0) & (coords_xyz[:, 1] < grid_size[1])
        & (coords_xyz[:, 2] >= 0) & (coords_xyz[:, 2] < grid_size[2])
    )
    coords_xyz = coords_xyz[valid]
    points = points[valid]
    if points.shape[0] == 0:
        return (
            np.zeros((1, feature_dim), dtype=np.float32),
            np.zeros((1, 3), dtype=np.int32),
        )

    unique_coords, inverse, counts = np.unique(
        coords_xyz, axis=0, return_inverse=True, return_counts=True
    )
    if unique_coords.shape[0] > max_voxels:
        keep = np.sort(
            np.random.choice(unique_coords.shape[0], max_voxels, replace=False)
        )
        old_to_new = -np.ones(unique_coords.shape[0], dtype=np.int32)
        old_to_new[keep] = np.arange(keep.shape[0], dtype=np.int32)
        point_keep = old_to_new[inverse] >= 0
        inverse = old_to_new[inverse[point_keep]]
        points = points[point_keep]
        unique_coords = unique_coords[keep]
        counts = np.bincount(inverse, minlength=unique_coords.shape[0])

    num_voxels = unique_coords.shape[0]
    sum_xyz = np.zeros((num_voxels, 3), dtype=np.float32)
    np.add.at(sum_xyz, inverse, points[:, :3].astype(np.float32))
    counts_f = counts.reshape(-1, 1).astype(np.float32)
    mean_xyz = sum_xyz / np.maximum(counts_f, 1.0)

    mean_intensity = None
    if include_intensity:
        if points.shape[1] < 4:
            raise ValueError("include_intensity=True requires a fourth point channel")
        sum_intensity = np.zeros((num_voxels, 1), dtype=np.float32)
        np.add.at(sum_intensity, inverse, points[:, 3:4].astype(np.float32))
        mean_intensity = sum_intensity / np.maximum(counts_f, 1.0)

    voxel_centers = pc_range[:3] + (
        unique_coords.astype(np.float32) + 0.5
    ) * voxel_size
    relative_xyz = (mean_xyz - voxel_centers) / voxel_size
    density = np.clip(np.log1p(counts_f) / np.log(64.0), 0.0, 1.0)

    parts = [relative_xyz]
    if mean_intensity is not None:
        parts.append(mean_intensity)
    parts.append(density)
    if use_abs_xyz:
        absolute_xyz = (mean_xyz - pc_range[:3]) / (
            pc_range[3:] - pc_range[:3] + 1e-6
        )
        parts.append(absolute_xyz * 2.0 - 1.0)

    voxel_features = np.concatenate(parts, axis=1).astype(np.float32)
    coords_zyx = unique_coords[:, [2, 1, 0]].astype(np.int32)
    return voxel_features, coords_zyx
