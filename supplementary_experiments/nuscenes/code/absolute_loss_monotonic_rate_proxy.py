#!/usr/bin/env python3
"""Six absolute task-loss heads plus a structurally monotonic six-rate head.

The shared sparse backbone and per-level hidden loss layers intentionally keep
the existing SparseCostProxyNet architecture.  Only the prediction semantics
change: every quantization level has an independent absolute task-loss output,
while log(1 + BPP) is decoded from positive adjacent increments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from train_cost_proxy import SparseCostProxyNet


NUM_LEVELS = 6
LEGACY_LOSS_HEAD_LEVEL_ORDER = (4, 3, 2, 1, 0)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def extract_model_state(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract a model state dict from old loss-only or rate-aware checkpoints."""
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return _strip_module_prefix(value)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _strip_module_prefix(checkpoint)
    raise KeyError("Checkpoint has no model/model_state/state_dict tensor mapping")


class AbsoluteLossMonotonicRateProxy(nn.Module):
    """Existing sparse backbone with six absolute-loss and monotonic-rate heads.

    Loss outputs are independent across levels and therefore may be non-monotonic.
    Each absolute loss is positive and restored to task units with a train-only
    per-level scale.  BPP is monotonic by construction in log1p space.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        feat_dim: int,
        loss_scales: Iterable[float],
        mean_log_bpp: Iterable[float],
        input_channels: int = 7,
    ) -> None:
        super().__init__()
        self.base = SparseCostProxyNet(
            input_channels=input_channels,
            spatial_shape=list(spatial_shape),
            feat_dim=feat_dim,
            num_cost_heads=NUM_LEVELS,
            num_targets=1,
            cost_nonnegative=False,
            monotonic_cost=False,
        )
        self.rate_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(feat_dim, NUM_LEVELS),
        )

        loss_scales_tensor = torch.as_tensor(list(loss_scales), dtype=torch.float32)
        mean_log_bpp_tensor = torch.as_tensor(list(mean_log_bpp), dtype=torch.float32)
        if loss_scales_tensor.shape != (NUM_LEVELS,):
            raise ValueError(f"loss_scales must have shape (6,), got {loss_scales_tensor.shape}")
        if mean_log_bpp_tensor.shape != (NUM_LEVELS,):
            raise ValueError(f"mean_log_bpp must have shape (6,), got {mean_log_bpp_tensor.shape}")
        if torch.any(loss_scales_tensor <= 0):
            raise ValueError("Every train-only loss scale must be positive")
        if torch.any(torch.diff(mean_log_bpp_tensor) < 0):
            raise ValueError("Training mean log-BPP must be ordered coarse to fine")

        mean_log_increments = torch.diff(
            torch.cat([mean_log_bpp_tensor.new_zeros(1), mean_log_bpp_tensor])
        ).clamp_min(1e-6)
        self.register_buffer("loss_scales", loss_scales_tensor)
        self.register_buffer("mean_log_increments", mean_log_increments)
        self.register_buffer(
            "unit_softplus_bias",
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32),
        )

        self._global_feature = None
        self.base.global_mlp.register_forward_hook(self._capture_global_feature)
        self.reset_new_loss_outputs()

    def _capture_global_feature(self, _module, _inputs, output) -> None:
        self._global_feature = output

    def reset_new_loss_outputs(self) -> None:
        """Initialize absolute outputs near each training-set loss scale."""
        bias = float(self.unit_softplus_bias)
        for head in self.base.cost_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(head[-1].bias, bias)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        self._global_feature = None
        base_output = self.base(voxel_features, voxel_coords, batch_size)
        if self._global_feature is None:
            raise RuntimeError("Sparse backbone global feature was not captured")

        raw_loss = base_output["cost_pred"].squeeze(-1)
        loss_pred = F.softplus(raw_loss) * self.loss_scales[None, :]

        raw_rate = self.rate_head(self._global_feature)
        positive_log_increments = self.mean_log_increments[None, :] * F.softplus(
            raw_rate + self.unit_softplus_bias
        )
        rate_log_pred = torch.cumsum(positive_log_increments, dim=1)
        bpp_pred = torch.expm1(rate_log_pred).clamp_min(0.0)
        return {
            "loss_raw": raw_loss,
            "loss_pred": loss_pred,
            "rate_raw": raw_rate,
            "rate_log_increments": positive_log_increments,
            "rate_log_pred": rate_log_pred,
            "bpp_pred": bpp_pred,
        }

    def forward_point_clouds(
        self,
        point_clouds: Sequence[torch.Tensor],
        voxel_size: Sequence[float] = (0.16, 0.16, 0.16),
        pc_range: Sequence[float] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        max_voxels: int = 50000,
        random_subsample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run geometry-only CUDA preprocessing and proxy prediction.

        Raw point tensors must already be on the proxy's CUDA device.  This is
        the default entry point for new monotonic training/evaluation code so
        it cannot silently fall back to the legacy NumPy voxelizer.
        """
        from gpu_voxelizer import voxelize_batch_gpu

        if random_subsample is None:
            random_subsample = bool(self.training)
        voxel_features, voxel_coords = voxelize_batch_gpu(
            point_clouds,
            voxel_size,
            pc_range,
            max_voxels,
            use_abs_xyz=True,
            include_intensity=False,
            random_subsample=random_subsample,
        )
        return self(voxel_features, voxel_coords, len(point_clouds))

    @torch.no_grad()
    def load_legacy_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Load compatible old weights without mapping delta finals to absolutes.

        The old five heads are ordered for levels (4,3,2,1,0).  Their first
        Linear layers are remapped to the corresponding natural level in the
        new six-head model.  Level 5 starts from a copy of level 4's hidden
        layer.  All six final loss Linear layers retain fresh initialization.
        """
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source = extract_model_state(checkpoint)
        target = self.state_dict()
        loaded: Dict[str, torch.Tensor] = {}
        copied_hidden = []

        for target_key, target_value in target.items():
            if "cost_heads" in target_key:
                continue
            if target_key in {"loss_scales", "mean_log_increments", "unit_softplus_bias"}:
                continue
            candidates = [target_key]
            if target_key.startswith("base."):
                candidates.append(target_key[len("base."):])
            for source_key in candidates:
                source_value = source.get(source_key)
                if source_value is not None and source_value.shape == target_value.shape:
                    loaded[target_key] = source_value
                    break

        for old_head, level in enumerate(LEGACY_LOSS_HEAD_LEVEL_ORDER):
            for suffix in ("0.weight", "0.bias"):
                target_key = f"base.cost_heads.{level}.{suffix}"
                source_candidates = (
                    f"base.cost_heads.{old_head}.{suffix}",
                    f"cost_heads.{old_head}.{suffix}",
                )
                for source_key in source_candidates:
                    source_value = source.get(source_key)
                    if source_value is not None and source_value.shape == target[target_key].shape:
                        loaded[target_key] = source_value
                        copied_hidden.append((source_key, target_key))
                        break

        # The new finest-level head has no legacy counterpart.  Start its hidden
        # representation from the closest old level (level 4), not from scratch.
        for suffix in ("0.weight", "0.bias"):
            source_candidates = (
                f"base.cost_heads.0.{suffix}",
                f"cost_heads.0.{suffix}",
            )
            target_key = f"base.cost_heads.5.{suffix}"
            for source_key in source_candidates:
                source_value = source.get(source_key)
                if source_value is not None and source_value.shape == target[target_key].shape:
                    loaded[target_key] = source_value.clone()
                    copied_hidden.append((source_key, target_key))
                    break

        current = self.state_dict()
        current.update(loaded)
        self.load_state_dict(current)
        final_loss_keys = [
            f"base.cost_heads.{level}.3.{name}"
            for level in range(NUM_LEVELS)
            for name in ("weight", "bias")
        ]
        return {
            "checkpoint": str(checkpoint_path),
            "loaded_tensor_count": len(loaded),
            "target_tensor_count": len(target),
            "copied_loss_hidden_layers": copied_hidden,
            "reinitialized_absolute_loss_outputs": final_loss_keys,
            "rate_head_loaded": all(
                key in loaded for key in target if key.startswith("rate_head.")
            ),
        }


def rd_levels(
    loss_by_level: torch.Tensor,
    bpp_by_level: torch.Tensor,
    lambdas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic routing: argmin_q L_q(x) + lambda R_q(x)."""
    scores = (
        loss_by_level[:, None, :]
        + lambdas[None, :, None] * bpp_by_level[:, None, :]
    )
    return scores.argmin(dim=-1), scores


def rate_monotonic_violation_rate(bpp_pred: torch.Tensor) -> torch.Tensor:
    return (torch.diff(bpp_pred, dim=1) < 0).float().mean()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "shared_sparse_backbone_and_global_mlp": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("base.") and ".cost_heads." not in name
        ),
        "six_absolute_loss_heads": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".cost_heads." in name
        ),
        "monotonic_bpp_head": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("rate_head.")
        ),
    }
