#!/usr/bin/env python3
"""LRProxy router for normalized small point clouds.

Only the feature backbone is changed.  The 256-dimensional shared feature,
six independent absolute-loss heads, monotonic six-rate BPP head, output keys,
and analytical RD selection inputs remain fixed across LRProxy variants.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn


NUM_LEVELS = 6


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def _extract_model_state(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return _strip_module_prefix(value)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _strip_module_prefix(checkpoint)
    raise KeyError("Checkpoint has no model/model_state/state_dict tensor mapping")


class LRProxyCostBackbone(nn.Module):
    """Permutation-invariant dense point encoder with no neighborhood index."""

    def __init__(self, input_channels: int = 3, feat_dim: int = 256,
                 num_cost_heads: int = NUM_LEVELS) -> None:
        super().__init__()
        if feat_dim != 256:
            raise ValueError("feat_dim must remain 256 for output-head compatibility")
        self.input_channels = int(input_channels)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Max captures salient geometry; mean captures density/distribution.
        self.global_mlp = nn.Sequential(
            nn.Linear(256, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        # Shared unchanged output-head design across LRProxy variants.
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(feat_dim, 1),
            )
            for _ in range(num_cost_heads)
        ])

    def forward(self, points: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] < self.input_channels:
            raise ValueError("points must have shape [B,N,C] with sufficient channels")
        x = self.point_mlp(points[..., :self.input_channels].transpose(1, 2).contiguous())
        if valid_mask is None:
            pooled_max = x.amax(dim=2)
            pooled_mean = x.mean(dim=2)
        else:
            if valid_mask.shape != points.shape[:2]:
                raise ValueError("valid_mask must have shape [B,N]")
            mask = valid_mask[:, None, :].to(dtype=torch.bool, device=x.device)
            pooled_max = x.masked_fill(~mask, torch.finfo(x.dtype).min).amax(dim=2)
            weights = mask.to(dtype=x.dtype)
            pooled_mean = (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        global_feature = self.global_mlp(torch.cat([pooled_max, pooled_mean], dim=1))
        raw_cost = torch.stack([head(global_feature) for head in self.cost_heads], dim=1)
        return {"cost_pred": raw_cost, "global_feature": global_feature}


class LRProxyBase(nn.Module):
    """LRProxy backbone plus unchanged direct-six-loss and monotonic-BPP heads."""

    def __init__(self, feat_dim: int, loss_scales: Iterable[float],
                 mean_log_bpp: Iterable[float], input_channels: int = 3) -> None:
        super().__init__()
        self.base = LRProxyCostBackbone(input_channels, feat_dim, NUM_LEVELS)
        # Shared unchanged output-head design across LRProxy variants.
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
            raise ValueError("Every loss scale must be positive")
        if torch.any(torch.diff(mean_log_bpp_tensor) < 0):
            raise ValueError("mean_log_bpp must be ordered coarse to fine")
        increments = torch.diff(torch.cat([mean_log_bpp_tensor.new_zeros(1), mean_log_bpp_tensor])).clamp_min(1e-6)
        self.register_buffer("loss_scales", loss_scales_tensor)
        self.register_buffer("mean_log_increments", increments)
        self.register_buffer("unit_softplus_bias", torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
        self.reset_new_loss_outputs()

    def reset_new_loss_outputs(self) -> None:
        bias = float(self.unit_softplus_bias)
        for head in self.base.cost_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(head[-1].bias, bias)

    def forward(self, points: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        base_output = self.base(points, valid_mask)
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

    @torch.no_grad()
    def load_compatible_heads(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load shape-compatible loss/rate head tensors from a legacy checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source = _extract_model_state(checkpoint)
        target = self.state_dict()
        loaded = {}
        skipped = []
        for key, value in source.items():
            if not (key.startswith("base.cost_heads.") or key.startswith("rate_head.")):
                continue
            if key in target and target[key].shape == value.shape:
                loaded[key] = value
            else:
                skipped.append(key)
        current = self.state_dict()
        current.update(loaded)
        self.load_state_dict(current)
        return {
            "checkpoint": str(checkpoint_path),
            "loaded_head_tensor_count": len(loaded),
            "loaded_head_parameter_count": int(sum(target[key].numel() for key in loaded if key in dict(self.named_parameters()))),
            "skipped_head_keys": skipped,
            "new_backbone_randomly_initialized": True,
        }

    @torch.no_grad()
    def load_full_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load a LRProxy checkpoint while keeping current-dataset scales."""
        loss_scales = self.loss_scales.detach().clone()
        mean_log_increments = self.mean_log_increments.detach().clone()
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source = _extract_model_state(checkpoint)
        self.load_state_dict(source, strict=True)
        # These buffers encode dataset target normalization, not learned weights.
        self.loss_scales.copy_(loss_scales)
        self.mean_log_increments.copy_(mean_log_increments)
        return {
            "checkpoint": str(checkpoint_path),
            "loaded_full_model": True,
            "loaded_tensor_count": len(source),
            "loaded_parameter_count": int(sum(value.numel() for value in self.parameters())),
            "current_dataset_output_scaling_preserved": True,
            "new_backbone_randomly_initialized": False,
        }


def count_parameters(model: LRProxyBase) -> Dict[str, int]:
    named = list(model.named_parameters())
    loss = sum(p.numel() for name, p in named if "cost_heads" in name)
    rate = sum(p.numel() for name, p in named if name.startswith("rate_head."))
    backbone = sum(p.numel() for name, p in named if "cost_heads" not in name and not name.startswith("rate_head."))
    return {
        "total": int(sum(p.numel() for _, p in named)),
        "trainable": int(sum(p.numel() for _, p in named if p.requires_grad)),
        "lrproxy_backbone": int(backbone),
        "six_absolute_loss_heads": int(loss),
        "monotonic_bpp_head": int(rate),
    }
