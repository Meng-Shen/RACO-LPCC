#!/usr/bin/env python3
"""Lite-S3 router: drop the expensive 256-channel sparse stage.

The six independent absolute-loss outputs and monotonic six-rate output are
unchanged.  Stem through stage3 and all 256-dimensional hidden heads remain
shape-compatible with the full legacy router.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from train_cost_proxy import SparseConvBlock, SparseCostProxyNet, SparseDownBlock, spconv


NUM_LEVELS = 6
LEGACY_LOSS_HEAD_LEVEL_ORDER = (4, 3, 2, 1, 0)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _extract_model_state(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return _strip_module_prefix(value)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _strip_module_prefix(checkpoint)
    raise KeyError("Checkpoint has no model/model_state/state_dict tensor mapping")


class LiteS3SparseCostBackbone(nn.Module):
    """Legacy sparse encoder truncated after the 128-channel stage3."""

    def __init__(
        self,
        input_channels: int,
        spatial_shape: Sequence[int],
        feat_dim: int = 256,
        num_cost_heads: int = NUM_LEVELS,
    ) -> None:
        super().__init__()
        self.spatial_shape = list(spatial_shape)
        self.stem = SparseConvBlock(input_channels, 32, indice_key="subm1")
        self.stage1 = nn.Sequential(
            SparseConvBlock(32, 32, indice_key="subm1a"),
            SparseConvBlock(32, 32, indice_key="subm1b"),
        )
        self.down2 = SparseDownBlock(32, 64, indice_key="spconv2")
        self.stage2 = nn.Sequential(
            SparseConvBlock(64, 64, indice_key="subm2a"),
            SparseConvBlock(64, 64, indice_key="subm2b"),
        )
        self.down3 = SparseDownBlock(64, 128, indice_key="spconv3")
        self.stage3 = nn.Sequential(
            SparseConvBlock(128, 128, indice_key="subm3a"),
            SparseConvBlock(128, 128, indice_key="subm3b"),
        )

        # Stage2 contributes 2*64 and stage3 contributes 2*128 pooled values.
        self.global_mlp = nn.Sequential(
            nn.Linear(384, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(feat_dim, 1),
            )
            for _ in range(num_cost_heads)
        ])

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size,
        )
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x_stage2 = x
        x = self.down3(x)
        x = self.stage3(x)
        x_stage3 = x

        pooled = torch.cat([
            SparseCostProxyNet.global_pool(
                stage.features, stage.indices[:, 0].long(), batch_size
            )
            for stage in (x_stage2, x_stage3)
        ], dim=1)
        global_feature = self.global_mlp(pooled)
        raw_cost = torch.stack(
            [head(global_feature) for head in self.cost_heads], dim=1
        )
        return {"cost_pred": raw_cost, "global_feature": global_feature}


class LiteS3AbsoluteLossMonotonicRateProxy(nn.Module):
    def __init__(
        self,
        spatial_shape: Sequence[int],
        feat_dim: int,
        loss_scales: Iterable[float],
        mean_log_bpp: Iterable[float],
        input_channels: int = 7,
    ) -> None:
        super().__init__()
        if int(feat_dim) != 256:
            raise ValueError("Lite-S3 keeps feat_dim=256 for legacy head compatibility")
        self.base = LiteS3SparseCostBackbone(
            input_channels=input_channels,
            spatial_shape=spatial_shape,
            feat_dim=feat_dim,
            num_cost_heads=NUM_LEVELS,
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
            raise ValueError("loss_scales must have shape (6,)")
        if mean_log_bpp_tensor.shape != (NUM_LEVELS,):
            raise ValueError("mean_log_bpp must have shape (6,)")
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
        self.reset_new_loss_outputs()

    def reset_new_loss_outputs(self) -> None:
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
        base_output = self.base(voxel_features, voxel_coords, batch_size)
        raw_loss = base_output["cost_pred"].squeeze(-1)
        loss_pred = F.softplus(raw_loss) * self.loss_scales[None, :]
        raw_rate = self.rate_head(base_output["global_feature"])
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
    def load_legacy_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source = _extract_model_state(checkpoint)
        target = self.state_dict()
        parameter_keys = set(dict(self.named_parameters()))
        loaded: Dict[str, torch.Tensor] = {}
        copied_hidden = []
        sliced_global_mlp = None

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
                if source_value is None:
                    continue
                if source_value.shape == target_value.shape:
                    loaded[target_key] = source_value
                    break
                if (
                    target_key == "base.global_mlp.0.weight"
                    and source_value.ndim == 2
                    and source_value.shape[0] == target_value.shape[0]
                    and source_value.shape[1] >= target_value.shape[1]
                ):
                    loaded[target_key] = source_value[:, : target_value.shape[1]].clone()
                    sliced_global_mlp = {
                        "source": source_key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                        "kept_columns": [0, target_value.shape[1]],
                    }
                    break

        for old_head, level in enumerate(LEGACY_LOSS_HEAD_LEVEL_ORDER):
            for suffix in ("0.weight", "0.bias"):
                target_key = "base.cost_heads.{}.{}".format(level, suffix)
                for source_key in (
                    "base.cost_heads.{}.{}".format(old_head, suffix),
                    "cost_heads.{}.{}".format(old_head, suffix),
                ):
                    source_value = source.get(source_key)
                    if source_value is not None and source_value.shape == target[target_key].shape:
                        loaded[target_key] = source_value
                        copied_hidden.append((source_key, target_key))
                        break
        for suffix in ("0.weight", "0.bias"):
            target_key = "base.cost_heads.5.{}".format(suffix)
            for source_key in (
                "base.cost_heads.0.{}".format(suffix),
                "cost_heads.0.{}".format(suffix),
            ):
                source_value = source.get(source_key)
                if source_value is not None and source_value.shape == target[target_key].shape:
                    loaded[target_key] = source_value.clone()
                    copied_hidden.append((source_key, target_key))
                    break

        current = self.state_dict()
        current.update(loaded)
        self.load_state_dict(current)
        loaded_parameter_count = sum(
            target[key].numel() for key in loaded if key in parameter_keys
        )
        total_parameter_count = sum(parameter.numel() for parameter in self.parameters())
        return {
            "checkpoint": str(checkpoint_path),
            "loaded_tensor_count": len(loaded),
            "target_tensor_count": len(target),
            "loaded_parameter_count": int(loaded_parameter_count),
            "total_parameter_count": int(total_parameter_count),
            "loaded_parameter_fraction": float(
                loaded_parameter_count / total_parameter_count
            ),
            "sliced_global_mlp": sliced_global_mlp,
            "copied_loss_hidden_layers": copied_hidden,
            "reinitialized_absolute_loss_outputs": [
                "base.cost_heads.{}.3.{}".format(level, name)
                for level in range(NUM_LEVELS)
                for name in ("weight", "bias")
            ],
            "rate_head_loaded": all(
                key in loaded for key in parameter_keys if key.startswith("rate_head.")
            ),
        }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "truncated_sparse_backbone_and_global_mlp": sum(
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
