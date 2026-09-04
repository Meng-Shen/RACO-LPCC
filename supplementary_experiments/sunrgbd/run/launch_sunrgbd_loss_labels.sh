#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
DATA="$ROOT/data/sunrgbd"
CONFIG="$CONFIG_DIR/votenet_sunrgbd_geometry_finetune.py"
CHECKPOINT="$ROOT/checkpoints/votenet_geometry_finetuned_best.pth"
OUT="$ROOT/labels/detector_loss"
SHARDS=7

mkdir -p "$OUT/train" "$OUT/val" "$ROOT/logs/loss_labels"
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  (
    mkdir -p "$OUT/train/shard_$shard" "$OUT/val/shard_$shard"
    CUDA_VISIBLE_DEVICES="$shard" "$PY" "$CODE/export_sunrgbd_quant_loss.py" \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA" \
      --split train --output "$OUT/train/shard_$shard/loss.csv" \
      --shard-id "$shard" --num-shards "$SHARDS" --device cuda:0
    CUDA_VISIBLE_DEVICES="$shard" "$PY" "$CODE/export_sunrgbd_quant_loss.py" \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA" \
      --split val --output "$OUT/val/shard_$shard/loss.csv" \
      --shard-id "$shard" --num-shards "$SHARDS" --device cuda:0
  ) > "$ROOT/logs/loss_labels/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" "$CODE/merge_sunrgbd_loss_shards.py" \
  --root "$OUT/train" --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
  --expected-scenes 5285 --output "$OUT/sunrgbd_train_absolute_losses.csv"
"$PY" "$CODE/merge_sunrgbd_loss_shards.py" \
  --root "$OUT/val" --split-file "$DATA/sunrgbd_trainval/val_data_idx.txt" \
  --expected-scenes 5050 --output "$OUT/sunrgbd_val_absolute_losses.csv"

date -Is > "$ROOT/state/LOSS_LABELS_COMPLETE"
