#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
TORCHRUN=/home/sm/miniconda3/envs/openmmlab/bin/torchrun
MMDET="$ROOT/mmdetection3d"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
DATA="$ROOT/data/sunrgbd"
CONFIG="$CONFIG_DIR/votenet_sunrgbd_geometry_finetune.py"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
STAGE=waiting_for_data

mkdir -p "$STATE" "$LOGS" "$ROOT/experiments" "$ROOT/checkpoints"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/PIPELINE_FAILED.json"' ERR
date -Is > "$STATE/PIPELINE_RECOVERY_STARTED"

while [[ ! -f "$STATE/PREPARE_COMPLETE" ]]; do sleep 30; done

STAGE=python_compile
cd "$CODE"
"$PY" -m py_compile ./*.py

STAGE=votenet_smoke
if [[ ! -f "$STATE/VOTENET_SMOKE_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$MMDET" "$PY" "$CODE/smoke_votenet_geometry.py" \
    > "$LOGS/votenet_smoke.log" 2>&1
fi

STAGE=quantization_probe
if [[ ! -f "$STATE/QUANT_PROBE_COMPLETE" ]]; then
  mkdir -p "$ROOT/experiments/quant_probe"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/export_sunrgbd_quant_loss.py" \
    --config "$CONFIG" \
    --checkpoint "$ROOT/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth" \
    --data-root "$DATA" --split val \
    --output "$ROOT/experiments/quant_probe/loss.csv" \
    --max-scenes 12 --device cuda:0 \
    > "$LOGS/quant_probe.log" 2>&1
  "$PY" "$CODE/analyze_sunrgbd_quant_probe.py" \
    "$ROOT/experiments/quant_probe/loss.csv" >> "$LOGS/quant_probe.log" 2>&1
  date -Is > "$STATE/QUANT_PROBE_COMPLETE"
fi

STAGE=gpcc_background
if [[ ! -f "$STATE/GPCC_COMPLETE" ]] && \
   ! pgrep -f "$SCRIPT_DIR/launch_sunrgbd_gpcc.sh" >/dev/null; then
  nohup "$SCRIPT_DIR/launch_sunrgbd_gpcc.sh" > "$LOGS/gpcc_pipeline.log" 2>&1 < /dev/null &
fi

STAGE=votenet_geometry_finetune
if [[ ! -f "$STATE/VOTENET_FINETUNE_COMPLETE" ]]; then
  cd "$MMDET"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$MMDET" \
    "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29677 \
    tools/train.py "$CONFIG" --launcher pytorch \
    > "$LOGS/votenet_geometry_finetune.log" 2>&1

  STAGE=select_votenet_checkpoint
  BEST_DETECTOR=$(find "$ROOT/experiments/votenet_geometry_finetune" -maxdepth 1 \
    -type f -name 'best*.pth' | sort | tail -1)
  if [[ -z "$BEST_DETECTOR" ]]; then
    BEST_DETECTOR=$(find "$ROOT/experiments/votenet_geometry_finetune" -maxdepth 1 \
      -type f -name 'epoch_*.pth' | sort -V | tail -1)
  fi
  test -n "$BEST_DETECTOR"
  test -f "$BEST_DETECTOR"
  cp "$BEST_DETECTOR" "$ROOT/checkpoints/votenet_geometry_finetuned_best.pth"
  printf '%s\n' "$BEST_DETECTOR" > "$STATE/VOTENET_SELECTED_CHECKPOINT.txt"
  date -Is > "$STATE/VOTENET_FINETUNE_COMPLETE"
fi

STAGE=detector_loss_labels
if [[ ! -f "$STATE/LOSS_LABELS_COMPLETE" ]]; then
  "$SCRIPT_DIR/launch_sunrgbd_loss_labels.sh" > "$LOGS/loss_labels_pipeline.log" 2>&1
fi

STAGE=wait_gpcc
while [[ ! -f "$STATE/GPCC_COMPLETE" ]]; do sleep 30; done

COMMON_ROUTER_ARGS=(
  --points-dir "$DATA/points"
  --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt"
  --loss-csv "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv"
  --bpp-csv "$ROOT/labels/gpcc/sunrgbd_train_gpcc.csv"
  --init-checkpoint "/home/sm/raco_rate_aware_nuscenes_20260822/experiments/nuscenes_sixloss_monotonic_lite_s3_20260826/best_map_bpp.pth"
)

STAGE=lite_s3_smoke
if [[ ! -f "$STATE/LITE_S3_SMOKE_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE" "$PY" "$CODE/train_sunrgbd_lite_s3_router_ddp.py" \
    "${COMMON_ROUTER_ARGS[@]}" \
    --output-dir "$ROOT/experiments/lite_s3_router_smoke" \
    --epochs 1 --patience 1 --batch-size 2 --workers 0 --max-scenes 16 \
    > "$LOGS/lite_s3_smoke.log" 2>&1
  date -Is > "$STATE/LITE_S3_SMOKE_COMPLETE"
fi

STAGE=lite_s3_train
if [[ ! -f "$STATE/LITE_S3_TRAIN_COMPLETE" ]]; then
  cd "$CODE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$CODE" \
    "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29678 \
    "$CODE/train_sunrgbd_lite_s3_router_ddp.py" \
    "${COMMON_ROUTER_ARGS[@]}" \
    --output-dir "$ROOT/experiments/lite_s3_router" \
    --epochs 30 --patience 7 --batch-size 4 --workers 2 \
    > "$LOGS/lite_s3_train.log" 2>&1
  date -Is > "$STATE/LITE_S3_TRAIN_COMPLETE"
fi

STAGE=router_prediction_and_lambda_calibration
if [[ ! -f "$STATE/ROUTER_PREDICTION_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE" "$PY" \
    "$CODE/predict_and_calibrate_sunrgbd_router.py" \
    --points-dir "$DATA/points" \
    --train-split "$DATA/sunrgbd_trainval/train_data_idx.txt" \
    --val-split "$DATA/sunrgbd_trainval/val_data_idx.txt" \
    --train-loss-csv "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv" \
    --train-bpp-csv "$ROOT/labels/gpcc/sunrgbd_train_gpcc.csv" \
    --val-loss-csv "$ROOT/labels/detector_loss/sunrgbd_val_absolute_losses.csv" \
    --val-bpp-csv "$ROOT/labels/gpcc/sunrgbd_val_gpcc.csv" \
    --checkpoint "$ROOT/experiments/lite_s3_router/best.pth" \
    --output-dir "$ROOT/experiments/lite_s3_router_eval" \
    > "$LOGS/router_predict_calibrate.log" 2>&1
  date -Is > "$STATE/ROUTER_PREDICTION_COMPLETE"
fi

STAGE=detector_prediction_cache
if [[ ! -f "$STATE/PREDICTIONS_COMPLETE" ]]; then
  "$SCRIPT_DIR/launch_sunrgbd_prediction_cache.sh" \
    > "$LOGS/prediction_cache_pipeline.log" 2>&1
fi

STAGE=official_ap_bpp_evaluation
if [[ ! -f "$STATE/PIPELINE_COMPLETE" ]]; then
  PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/evaluate_plot_sunrgbd_ap_bpp.py" \
    --prediction-root "$ROOT/predictions/val_six_levels" \
    --router-csv "$ROOT/experiments/lite_s3_router_eval/val_router_predictions.csv" \
    --gpcc-csv "$ROOT/labels/gpcc/sunrgbd_val_gpcc.csv" \
    --lambda-json "$ROOT/experiments/lite_s3_router_eval/lambda_calibration_and_metrics.json" \
    --output-dir "$ROOT/results/lite_s3_vs_gpcc" \
    > "$LOGS/final_evaluation.log" 2>&1
fi

STAGE=complete
date -Is > "$STATE/PIPELINE_COMPLETE"
rm -f "$STATE/PIPELINE_FAILED.json"
