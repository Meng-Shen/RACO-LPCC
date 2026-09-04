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
OUT="$ROOT/final_qsteps_160_120_100_80_60_40/val_predictions"
LOGS="$ROOT/logs/final_val_predictions"
SHARDS=7

mkdir -p "$OUT" "$LOGS"
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  (
    mkdir -p "$OUT/shard_$shard"
    CUDA_VISIBLE_DEVICES="$shard" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONPATH="$ROOT/mmdetection3d:$CODE" \
      "$PY" "$CODE/export_sunrgbd_quant_predictions.py" \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA" \
      --split val --qsteps-mm 160 120 100 80 60 40 \
      --output "$OUT/shard_$shard/predictions.pkl" \
      --shard-id "$shard" --num-shards "$SHARDS" --device cuda:0
  ) > "$LOGS/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$ROOT/state/FINAL_QSTEPS_VAL_PREDICTIONS_COMPLETE"
