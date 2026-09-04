#!/usr/bin/env python3
from train_scannet_rate_aware_proxy import load_labels


train = load_labels(
    "labels/train_quant_losses/train_losses_merged.csv",
    "labels/nuscenes_train_gpcc_per_frame_per_rate.csv",
    "nuscenes",
)
val = load_labels(
    "labels/val_quant_losses/val_losses_merged.csv",
    "labels/nuscenes_val_gpcc_per_frame_per_rate.csv",
    "nuscenes",
)
print(f"train={len(train)} val={len(val)}")
