#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
MMDET="$ROOT/mmdetection3d"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
RESULTS="$ROOT/results/oracle_vs_gpcc_train_preview"
STAGE=merge_completed_train_loss

mkdir -p "$STATE" "$LOGS" "$RESULTS"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/TRAIN_ORACLE_BASELINE_FAILED.json"' ERR
date -Is > "$STATE/TRAIN_ORACLE_BASELINE_PREVIEW_STARTED"

if [[ ! -f "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv" ]]; then
  "$PY" "$CODE/merge_sunrgbd_loss_shards.py" \
    --root "$ROOT/labels/detector_loss/train" \
    --split-file "$ROOT/data/sunrgbd/sunrgbd_trainval/train_data_idx.txt" \
    --expected-scenes 5285 \
    --output "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv"
fi

STAGE=train_detector_prediction_cache
if [[ ! -f "$STATE/TRAIN_PREDICTIONS_COMPLETE" ]]; then
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$SCRIPT_DIR/launch_sunrgbd_train_prediction_cache.sh" \
    > "$LOGS/train_oracle_prediction_cache_pipeline.log" 2>&1
fi

STAGE=train_oracle_and_baseline_evaluation
PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/evaluate_plot_sunrgbd_oracle_gpcc.py" \
  --prediction-root "$ROOT/predictions/train_six_levels" \
  --loss-csv "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv" \
  --gpcc-csv "$ROOT/labels/gpcc/sunrgbd_train_gpcc.csv" \
  --split-name train --expected-scenes 5285 \
  --output-dir "$RESULTS" \
  > "$LOGS/train_oracle_baseline_evaluation.log" 2>&1

STAGE=complete
date -Is > "$STATE/TRAIN_ORACLE_BASELINE_COMPLETE"
rm -f "$STATE/TRAIN_ORACLE_BASELINE_FAILED.json"
