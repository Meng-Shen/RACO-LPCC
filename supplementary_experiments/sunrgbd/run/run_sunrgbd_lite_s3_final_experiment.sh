#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
TORCHRUN=/home/sm/miniconda3/envs/openmmlab/bin/torchrun
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
MMDET="$ROOT/mmdetection3d"
DATA="$ROOT/data/sunrgbd"
FINAL="$ROOT/final_qsteps_160_120_100_80_60_40"
STATE="$ROOT/state"
EXP="$ROOT/experiments/lite_s3_final_qsteps_160_120_100_80_60_40_full5285_20260828"
ROUTING="$ROOT/experiments/lite_s3_final_qsteps_160_120_100_80_60_40_full5285_eval_20260828"
RESULTS="$ROOT/results/lite_s3_vs_gpcc_final_qsteps_160_120_100_80_60_40_full5285_20260828"
LOGS="$ROOT/logs/lite_s3_final_qsteps_160_120_100_80_60_40_full5285_20260828"
INIT=/home/sm/raco_rate_aware_nuscenes_20260822/experiments/nuscenes_sixloss_monotonic_lite_s3_20260826/best_map_bpp.pth
STAGE=initializing

mkdir -p "$STATE" "$EXP" "$ROUTING" "$RESULTS" "$LOGS"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/SUNRGBD_LITE_S3_FINAL_FAILED.json"' ERR
date -Is > "$STATE/SUNRGBD_LITE_S3_FINAL_STARTED"

STAGE=wait_for_official_val_bpp
while [[ ! -f "$STATE/FINAL_QSTEPS_VAL_GPCC_COMPLETE" ]]; do sleep 20; done

STAGE=lite_s3_full_train
if [[ ! -f "$STATE/SUNRGBD_LITE_S3_FINAL_TRAIN_COMPLETE" ]]; then
  cd "$CODE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$CODE" \
    "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29718 \
    "$CODE/train_sunrgbd_lite_s3_router_ddp.py" \
    --points-dir "$DATA/points" \
    --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
    --loss-csv "$FINAL/sunrgbd_train_absolute_losses.csv" \
    --bpp-csv "$FINAL/sunrgbd_train_gpcc.csv" \
    --init-checkpoint "$INIT" --output-dir "$EXP" \
    --epochs 30 --patience 7 --batch-size 4 --workers 2 \
    --voxel-size 0.16 0.16 0.16 \
    > "$LOGS/train.log" 2>&1
  date -Is > "$STATE/SUNRGBD_LITE_S3_FINAL_TRAIN_COMPLETE"
fi

STAGE=train_calibration_and_test_routing
if [[ ! -f "$STATE/SUNRGBD_LITE_S3_FINAL_ROUTING_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$CODE" \
    "$PY" "$CODE/predict_and_calibrate_sunrgbd_router.py" \
    --points-dir "$DATA/points" \
    --train-split "$DATA/sunrgbd_trainval/train_data_idx.txt" \
    --val-split "$DATA/sunrgbd_trainval/val_data_idx.txt" \
    --train-loss-csv "$FINAL/sunrgbd_train_absolute_losses.csv" \
    --train-bpp-csv "$FINAL/sunrgbd_train_gpcc.csv" \
    --val-bpp-csv "$FINAL/sunrgbd_val_gpcc.csv" \
    --checkpoint "$EXP/best.pth" --output-dir "$ROUTING" \
    --batch-size 8 --workers 2 \
    > "$LOGS/router_predict_calibrate.log" 2>&1
  date -Is > "$STATE/SUNRGBD_LITE_S3_FINAL_ROUTING_COMPLETE"
fi

STAGE=official_val_detector_prediction_cache
if [[ ! -f "$STATE/FINAL_QSTEPS_VAL_PREDICTIONS_COMPLETE" ]]; then
  "$SCRIPT_DIR/launch_sunrgbd_final_val_prediction_cache.sh" \
    > "$LOGS/val_prediction_cache.log" 2>&1
fi

STAGE=official_val_ap_bpp_evaluation
if [[ ! -f "$STATE/SUNRGBD_LITE_S3_FINAL_EVALUATION_COMPLETE" ]]; then
  PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/evaluate_plot_sunrgbd_ap_bpp.py" \
    --prediction-root "$FINAL/val_predictions" \
    --router-csv "$ROUTING/val_router_predictions.csv" \
    --gpcc-csv "$FINAL/sunrgbd_val_gpcc.csv" \
    --lambda-json "$ROUTING/lambda_calibration_and_metrics.json" \
    --output-dir "$RESULTS" \
    > "$LOGS/evaluation.log" 2>&1
  date -Is > "$STATE/SUNRGBD_LITE_S3_FINAL_EVALUATION_COMPLETE"
fi

STAGE=complete
date -Is > "$STATE/SUNRGBD_LITE_S3_FINAL_COMPLETE"
rm -f "$STATE/SUNRGBD_LITE_S3_FINAL_FAILED.json"
