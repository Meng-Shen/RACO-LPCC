#!/usr/bin/env python3
"""LRproxy: pure-XYZ LRProxy router with unchanged output heads."""

from __future__ import annotations

import torch

from lrproxy_base import (
    NUM_LEVELS,
    LRProxyBase,
    count_parameters,
)


ABSOLUTE_XYZ_CHANNELS = (4, 5, 6)


class LRProxy(
    LRProxyBase
):
    """Use only normalized voxel-mean absolute XYZ; keep all output heads."""

    alias = "LRproxy"
    feature_semantics = "normalized voxel-mean absolute XYZ (3)"

    def __init__(self, feat_dim, loss_scales, mean_log_bpp):
        super().__init__(
            feat_dim=feat_dim,
            loss_scales=loss_scales,
            mean_log_bpp=mean_log_bpp,
            input_channels=3,
        )

def select_xyz_features(voxel_features: torch.Tensor) -> torch.Tensor:
    """Select normalized absolute XYZ from [rel_xyz, density, abs_xyz_norm]."""
    if voxel_features.ndim != 2 or voxel_features.shape[1] != 7:
        raise ValueError(
            f"Expected flattened voxel features [M,7], got {voxel_features.shape}"
        )
    return voxel_features[:, ABSOLUTE_XYZ_CHANNELS].contiguous()


__all__ = [
    "NUM_LEVELS",
    "LRProxy",
    "ABSOLUTE_XYZ_CHANNELS",
    "select_xyz_features",
    "count_parameters",
]
