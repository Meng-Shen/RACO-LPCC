#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
MMDET="$ROOT/mmdetection3d"
DATA="$ROOT/data/sunrgbd"
CONFIG="$CONFIG_DIR/votenet_sunrgbd_geometry_finetune.py"
CHECKPOINT="$ROOT/checkpoints/votenet_geometry_finetuned_best.pth"
STATE="$ROOT/state"
LOGS="$ROOT/logs/revised_qsteps_160_120_80_60_40_20"
EXTRA="$ROOT/supplemental/qsteps_120_60"
COMBINED="$ROOT/revised_qsteps_160_120_80_60_40_20"
RESULTS="$ROOT/results/oracle_vs_gpcc_train_preview_qsteps_160_120_80_60_40_20"
STAGE=initializing

mkdir -p "$STATE" "$LOGS" "$EXTRA" "$COMBINED" "$RESULTS"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/REVISED_QSTEPS_TRAIN_PREVIEW_FAILED.json"' ERR
date -Is > "$STATE/REVISED_QSTEPS_TRAIN_PREVIEW_STARTED"

run_loss() {
  local pids=()
  mkdir -p "$EXTRA/loss" "$LOGS/loss"
  for shard in $(seq 0 6); do
    (
      mkdir -p "$EXTRA/loss/shard_$shard"
      CUDA_VISIBLE_DEVICES="$shard" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "$PY" "$CODE/export_sunrgbd_quant_loss.py" \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA" \
        --split train --qsteps-mm 120 60 \
        --output "$EXTRA/loss/shard_$shard/loss.csv" \
        --shard-id "$shard" --num-shards 7 --device cuda:0
    ) > "$LOGS/loss/shard_$shard.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$CODE/merge_sunrgbd_loss_shards.py" \
    --root "$EXTRA/loss" \
    --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
    --expected-scenes 5285 --num-levels 2 \
    --output "$EXTRA/sunrgbd_train_absolute_losses_qsteps_120_60.csv"
}

run_gpcc() {
  local pids=()
  mkdir -p "$EXTRA/gpcc" "$EXTRA/gpcc_tmp" "$LOGS/gpcc"
  for shard in $(seq 0 13); do
    (
      mkdir -p "$EXTRA/gpcc/shard_$shard" "$EXTRA/gpcc_tmp/shard_$shard"
      "$PY" "$CODE/measure_sunrgbd_gpcc.py" \
        --points-dir "$DATA/points" \
        --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
        --split-name train --qsteps-mm 120 60 \
        --output "$EXTRA/gpcc/shard_$shard/gpcc.csv" \
        --tmp-dir "$EXTRA/gpcc_tmp/shard_$shard" \
        --tmc3 "$ROOT/bin/tmc3_v22" --config "$ROOT/bin/dense.cfg" \
        --shard-id "$shard" --num-shards 14
    ) > "$LOGS/gpcc/shard_$shard.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$CODE/measure_sunrgbd_gpcc.py" \
    --merge-root "$EXTRA/gpcc" \
    --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
    --expected-scenes 5285 --qsteps-mm 120 60 \
    --output "$EXTRA/sunrgbd_train_gpcc_qsteps_120_60.csv"
}

STAGE=supplemental_loss_and_gpcc
run_loss > "$LOGS/loss_pipeline.log" 2>&1 &
loss_pid=$!
run_gpcc > "$LOGS/gpcc_pipeline.log" 2>&1 &
gpcc_pid=$!
wait "$loss_pid"
wait "$gpcc_pid"
date -Is > "$STATE/REVISED_QSTEPS_LOSS_GPCC_COMPLETE"

STAGE=supplemental_predictions
mkdir -p "$EXTRA/predictions" "$LOGS/predictions"
pids=()
for shard in $(seq 0 6); do
  (
    mkdir -p "$EXTRA/predictions/shard_$shard"
    CUDA_VISIBLE_DEVICES="$shard" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONPATH="$MMDET:$CODE" \
      "$PY" "$CODE/export_sunrgbd_quant_predictions.py" \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA" \
      --split train --qsteps-mm 120 60 \
      --output "$EXTRA/predictions/shard_$shard/predictions.pkl" \
      --shard-id "$shard" --num-shards 7 --device cuda:0
  ) > "$LOGS/predictions/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$STATE/REVISED_QSTEPS_PREDICTIONS_COMPLETE"

STAGE=combining_cached_assets
PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/combine_sunrgbd_qstep_assets.py" \
  --old-loss "$ROOT/labels/detector_loss/sunrgbd_train_absolute_losses.csv" \
  --extra-loss "$EXTRA/sunrgbd_train_absolute_losses_qsteps_120_60.csv" \
  --output-loss "$COMBINED/sunrgbd_train_absolute_losses.csv" \
  --old-gpcc "$ROOT/labels/gpcc/sunrgbd_train_gpcc.csv" \
  --extra-gpcc "$EXTRA/sunrgbd_train_gpcc_qsteps_120_60.csv" \
  --output-gpcc "$COMBINED/sunrgbd_train_gpcc.csv" \
  --old-predictions "$ROOT/predictions/train_six_levels" \
  --extra-predictions "$EXTRA/predictions" \
  --output-predictions "$COMBINED/predictions" \
  > "$LOGS/combine.log" 2>&1

STAGE=evaluating_and_plotting
PYTHONPATH="$MMDET:$CODE" "$PY" "$CODE/evaluate_plot_sunrgbd_oracle_gpcc.py" \
  --prediction-root "$COMBINED/predictions" \
  --loss-csv "$COMBINED/sunrgbd_train_absolute_losses.csv" \
  --gpcc-csv "$COMBINED/sunrgbd_train_gpcc.csv" \
  --calibration-loss-csv "$COMBINED/sunrgbd_train_absolute_losses.csv" \
  --calibration-gpcc-csv "$COMBINED/sunrgbd_train_gpcc.csv" \
  --calibration-split-name train --split-name train \
  --expected-scenes 5285 --output-dir "$RESULTS" \
  > "$LOGS/evaluation.log" 2>&1

STAGE=complete
date -Is > "$STATE/REVISED_QSTEPS_TRAIN_PREVIEW_COMPLETE"
rm -f "$STATE/REVISED_QSTEPS_TRAIN_PREVIEW_FAILED.json"
