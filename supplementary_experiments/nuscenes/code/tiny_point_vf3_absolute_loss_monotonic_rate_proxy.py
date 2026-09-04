#!/usr/bin/env python3
"""TinyPoint-VF3: three-coordinate ablation initialized from TinyPoint-VF7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from tiny_point_absolute_loss_monotonic_rate_proxy import (
    NUM_LEVELS,
    TinyPointAbsoluteLossMonotonicRateProxy,
    _extract_model_state,
    count_parameters,
)


VF7_ABSOLUTE_XYZ_CHANNELS = (4, 5, 6)


class TinyPointVF3AbsoluteLossMonotonicRateProxy(
    TinyPointAbsoluteLossMonotonicRateProxy
):
    """Use only normalized voxel-mean absolute XYZ; keep all output heads."""

    alias = "TinyPoint-VF3"
    feature_semantics = "normalized voxel-mean absolute XYZ (3)"

    def __init__(self, feat_dim, loss_scales, mean_log_bpp):
        super().__init__(
            feat_dim=feat_dim,
            loss_scales=loss_scales,
            mean_log_bpp=mean_log_bpp,
            input_channels=3,
        )

    @torch.no_grad()
    def load_from_vf7_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load all VF7 weights, slicing its normalized-absolute-XYZ channels."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source = _extract_model_state(checkpoint)
        target = self.state_dict()
        parameter_keys = set(dict(self.named_parameters()))
        preserve_current_buffers = {"loss_scales", "mean_log_increments"}
        first_key = "base.point_mlp.0.weight"
        loaded = {}
        mapped = []
        skipped = []
        for key, target_value in target.items():
            if key in preserve_current_buffers:
                continue
            if key == first_key:
                source_value = source.get(key)
                if source_value is None or tuple(source_value.shape) != (32, 7, 1):
                    raise RuntimeError(
                        f"VF7 first layer must be [32,7,1], got "
                        f"{None if source_value is None else tuple(source_value.shape)}"
                    )
                loaded[key] = source_value[:, VF7_ABSOLUTE_XYZ_CHANNELS, :].clone()
                mapped.append(key)
            elif key in source and source[key].shape == target_value.shape:
                loaded[key] = source[key]
            else:
                skipped.append(key)

        missing_parameters = sorted(parameter_keys.difference(loaded))
        if missing_parameters:
            raise RuntimeError(f"VF7-to-VF3 initialization missed parameters: {missing_parameters}")
        current = self.state_dict()
        current.update(loaded)
        self.load_state_dict(current, strict=True)
        return {
            "checkpoint": str(checkpoint_path),
            "source_alias": checkpoint.get("model_alias", "TinyPoint-VF7"),
            "loaded_parameter_tensor_count": len(parameter_keys),
            "loaded_parameter_count": int(sum(value.numel() for value in self.parameters())),
            "mapped_first_layer": first_key,
            "mapped_source_channels": list(VF7_ABSOLUTE_XYZ_CHANNELS),
            "mapped_feature_semantics": self.feature_semantics,
            "preserved_current_dataset_buffers": sorted(preserve_current_buffers),
            "skipped_nonparameter_tensors": skipped,
            "new_backbone_randomly_initialized": False,
        }


def select_vf3_features(vf7_features: torch.Tensor) -> torch.Tensor:
    """Select normalized absolute XYZ from [rel_xyz, density, abs_xyz_norm]."""
    if vf7_features.ndim != 2 or vf7_features.shape[1] != 7:
        raise ValueError(f"Expected flattened VF7 features [M,7], got {vf7_features.shape}")
    return vf7_features[:, VF7_ABSOLUTE_XYZ_CHANNELS].contiguous()


__all__ = [
    "NUM_LEVELS",
    "TinyPointVF3AbsoluteLossMonotonicRateProxy",
    "VF7_ABSOLUTE_XYZ_CHANNELS",
    "select_vf3_features",
    "count_parameters",
]
