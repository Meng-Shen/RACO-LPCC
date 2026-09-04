#!/usr/bin/env python3
"""Point-MAE ShapeNet classifier compatible with the official pretrain.pth.

The learned module names intentionally follow Pang-Yatian/Point-MAE so that
MAE_encoder weights load without approximation. FPS uses the already-installed
MMCV CUDA op; neighborhood search uses exact torch.cdist KNN.
"""

from __future__ import annotations

from pathlib import Path

import torch
from mmcv.ops import furthest_point_sample
from torch import nn


def drop_path(x, probability=0.0, training=False):
    if probability == 0.0 or not training:
        return x
    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    random.floor_()
    return x.div(keep) * random


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        return drop_path(x, self.probability, self.training)


class Encoder(nn.Module):
    def __init__(self, encoder_channel=384):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups):
        batch, groups, points, _ = point_groups.shape
        point_groups = point_groups.reshape(batch * groups, points, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        global_feature = feature.max(dim=2, keepdim=True).values
        feature = torch.cat([global_feature.expand(-1, -1, points), feature], dim=1)
        feature = self.second_conv(feature).max(dim=2).values
        return feature.reshape(batch, groups, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group=64, group_size=32):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, xyz):
        batch, num_points, _ = xyz.shape
        center_index = furthest_point_sample(xyz.contiguous(), self.num_group).long()
        center = torch.gather(xyz, 1, center_index[:, :, None].expand(-1, -1, 3))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            distance = torch.cdist(center.float(), xyz.float())
            index = distance.topk(self.group_size, dim=-1, largest=False, sorted=False).indices
        base = torch.arange(batch, device=xyz.device).view(-1, 1, 1) * num_points
        flat_index = (index + base).reshape(-1)
        neighborhood = xyz.reshape(batch * num_points, 3)[flat_index]
        neighborhood = neighborhood.reshape(batch, self.num_group, self.group_size, 3)
        return (neighborhood - center.unsqueeze(2)).contiguous(), center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.attn = Attention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, drop_path_rate=0.1):
        super().__init__()
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, drop_path_rate=rates[index])
            for index in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointMAEClassifier(nn.Module):
    def __init__(self, num_classes=55, trans_dim=384, depth=12, num_heads=6,
                 group_size=32, num_group=64, drop_path_rate=0.1):
        super().__init__()
        self.group_divider = Group(num_group, group_size)
        self.encoder = Encoder(trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim),
        )
        self.blocks = TransformerEncoder(trans_dim, depth, num_heads, drop_path_rate)
        self.norm = nn.LayerNorm(trans_dim)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, points):
        neighborhood, center = self.group_divider(points)
        group_tokens = self.encoder(neighborhood)
        cls_token = self.cls_token.expand(len(points), -1, -1)
        cls_pos = self.cls_pos.expand(len(points), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat([cls_token, group_tokens], dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)
        x = self.norm(self.blocks(x, pos))
        feature = torch.cat([x[:, 0], x[:, 1:].max(dim=1).values], dim=-1)
        return self.cls_head_finetune(feature)


def load_shapenet_pretrain(model: PointMAEClassifier, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("base_model", checkpoint)
    remapped = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[7:]
        if key.startswith("MAE_encoder."):
            key = key[len("MAE_encoder."):]
        elif key.startswith("base_model."):
            key = key[len("base_model."):]
        if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
            remapped[key] = value
    incompatible = model.load_state_dict(remapped, strict=False)
    required_prefixes = ("encoder.", "pos_embed.", "blocks.", "norm.")
    missing_backbone = [
        key for key in model.state_dict()
        if key.startswith(required_prefixes) and key not in remapped
    ]
    if missing_backbone:
        raise RuntimeError(f"ShapeNet pretrain is missing backbone keys: {missing_backbone[:20]}")
    return {
        "loaded_tensor_count": len(remapped),
        "loaded_keys": sorted(remapped),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
    }
