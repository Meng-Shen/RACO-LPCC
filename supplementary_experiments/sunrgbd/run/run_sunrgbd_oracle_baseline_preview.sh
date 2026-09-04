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
RESULTS="$ROOT/results/oracle_vs_gpcc_preview"
STAGE=waiting_for_loss_labels

mkdir -p "$STATE" "$LOGS" "$RESULTS"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/ORACLE_BASELINE_FAILED.json"' ERR
date -Is > "$STATE/ORACLE_BASELINE_PREVIEW_STARTED"

while [[ ! -f "$STATE/LOSS_LABELS_COMPLETE" ]]; do sleep 30; done

STAGE=detector_prediction_cache
if [[ ! -f "$STATE/PREDICTIONS_COMPLETE" ]]; then
  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$SCRIPT_DIR/launch_sunrgbd_prediction_cache.sh" \
    > "$LOGS/oracle_prediction_cache_pipeline.log" 2>&1
fi

STAGE=oracle_and_baseline_evaluation
PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/evaluate_plot_sunrgbd_oracle_gpcc.py" \
  --prediction-root "$ROOT/predictions/val_six_levels" \
  --train-loss-csv "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv" \
  --val-loss-csv "$ROOT/labels/detector_loss/sunrgbd_val_absolute_losses.csv" \
  --train-gpcc-csv "$ROOT/labels/gpcc/sunrgbd_train_gpcc.csv" \
  --val-gpcc-csv "$ROOT/labels/gpcc/sunrgbd_val_gpcc.csv" \
  --output-dir "$RESULTS" \
  > "$LOGS/oracle_baseline_evaluation.log" 2>&1

STAGE=complete
date -Is > "$STATE/ORACLE_BASELINE_COMPLETE"
rm -f "$STATE/ORACLE_BASELINE_FAILED.json"
