#!/usr/bin/env python3
"""TinyPoint-VF7: TinyPoint with Lite-S3-aligned seven-dimensional voxel features."""

from __future__ import annotations

from tiny_point_absolute_loss_monotonic_rate_proxy import (
    NUM_LEVELS,
    TinyPointAbsoluteLossMonotonicRateProxy,
    count_parameters,
)


class TinyPointVF7AbsoluteLossMonotonicRateProxy(
    TinyPointAbsoluteLossMonotonicRateProxy
):
    """Seven-channel alias; six loss heads and monotonic BPP head are unchanged."""

    alias = "TinyPoint-VF7"
    feature_semantics = (
        "voxel-relative XYZ (3), log-density (1), normalized absolute XYZ (3)"
    )

    def __init__(self, feat_dim, loss_scales, mean_log_bpp):
        super().__init__(
            feat_dim=feat_dim,
            loss_scales=loss_scales,
            mean_log_bpp=mean_log_bpp,
            input_channels=7,
        )


__all__ = [
    "NUM_LEVELS",
    "TinyPointVF7AbsoluteLossMonotonicRateProxy",
    "count_parameters",
]
