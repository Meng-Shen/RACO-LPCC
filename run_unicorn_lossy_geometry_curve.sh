#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPCDET_TOOLS="${SCRIPT_DIR}/OpenPCDet/tools"

DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
UNICORN_PYTHON_BIN="${UNICORN_PYTHON_BIN:-$PYTHON_BIN}"

UNICORN_ROOT="${UNICORN_ROOT:-/public/DATA/sm/Unicorn}"
CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_fov_geometry.yaml}"
DET_CKPT="${DET_CKPT:-ckpt/model_non_reflectance.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-${KITTI_ROOT}/ImageSets/train.txt}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/unicorn_fov}"

UNICORN_MODEL_DIR="${UNICORN_MODEL_DIR:-${SCRIPT_DIR}/unicorn/checkpoints}"
SR_CKPT="${SR_CKPT:-${UNICORN_MODEL_DIR}/sr/epoch_last.pth}"
OFFSET_CKPT="${OFFSET_CKPT:-${UNICORN_MODEL_DIR}/offset/epoch_last.pth}"
LOSSLESS_LOW_CKPT="${LOSSLESS_LOW_CKPT:-${UNICORN_MODEL_DIR}/lossless/epoch_last.pth}"

UNICORN_RESULTS_DIR="${UNICORN_RESULTS_DIR:-${OUT_DIR}/unicorn}"
UNICORN_TMP_DIR="${UNICORN_TMP_DIR:-${OUT_DIR}/tmp}"
UNICORN_BITSTREAM_DIR="${UNICORN_BITSTREAM_DIR:-${OUT_DIR}/bitstreams}"
UNICORN_DECODED_DIR="${UNICORN_DECODED_DIR:-${OUT_DIR}/decoded}"
AP_CSV="${AP_CSV:-${OUT_DIR}/unicorn_ap.csv}"
CURVE_CSV="${CURVE_CSV:-${OUT_DIR}/unicorn_full_curve.csv}"
RATE_CONFIG_CSV="${RATE_CONFIG_CSV:-${OUT_DIR}/unicorn_rate_points.csv}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_LOSSLESS="${RUN_LOSSLESS:-$RUN_TRAIN}"
RUN_SR="${RUN_SR:-$RUN_TRAIN}"
RUN_OFFSET="${RUN_OFFSET:-$RUN_TRAIN}"
RUN_UNICORN="${RUN_UNICORN:-1}"
RUN_AP="${RUN_AP:-1}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
UNICORN_RESUME="${UNICORN_RESUME:-1}"
GPU_GUARD="${GPU_GUARD:-1}"
GPU_GUARD_ID="${GPU_GUARD_ID:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_GUARD_ID="${GPU_GUARD_ID%%,*}"
GPU_GUARD_INTERVAL="${GPU_GUARD_INTERVAL:-1.0}"

TRAIN_POSQ="${TRAIN_POSQ:-64.0}"
EPOCHS="${EPOCHS:-100}"
TRAIN_NUM="${TRAIN_NUM:-1000000}"
TEST_NUM="${TEST_NUM:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_LR="${TRAIN_LR:-0.0001}"
MAX_STEPS="${MAX_STEPS:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
CHANNELS="${CHANNELS:-32}"
OFFSET_CHANNELS="${OFFSET_CHANNELS:-64}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
BLOCK_LAYERS="${BLOCK_LAYERS:-3}"
BLOCK_TYPE="${BLOCK_TYPE:-conv}"
LOSSLESS_STAGE="${LOSSLESS_STAGE:-8}"
LOSSLESS_SCALE="${LOSSLESS_SCALE:-5}"
RESOLUTION="${RESOLUTION:-80000}"
POSQUANTSCALE_LIST="${POSQUANTSCALE_LIST:-2 4 8 16 32}"
# Test rate points use "scale_AE:scale_SR:posQuantscale".
# This direct KITTI wrapper trains/evaluates lossless, SR, and offset modules, so scale_AE is kept at 0.
RATES="${RATES:-0:0:1,0:1:2,0:1:4,0:1:8,0:1:16,0:1:32}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$SCRIPT_DIR" "$1" ;;
  esac
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

OUT_DIR="$(abs_path "$OUT_DIR")"
UNICORN_MODEL_DIR="$(abs_path "$UNICORN_MODEL_DIR")"
SR_CKPT="$(abs_path "$SR_CKPT")"
OFFSET_CKPT="$(abs_path "$OFFSET_CKPT")"
LOSSLESS_LOW_CKPT="$(abs_path "$LOSSLESS_LOW_CKPT")"
UNICORN_RESULTS_DIR="$(abs_path "$UNICORN_RESULTS_DIR")"
UNICORN_TMP_DIR="$(abs_path "$UNICORN_TMP_DIR")"
UNICORN_BITSTREAM_DIR="$(abs_path "$UNICORN_BITSTREAM_DIR")"
UNICORN_DECODED_DIR="$(abs_path "$UNICORN_DECODED_DIR")"
AP_CSV="$(abs_path "$AP_CSV")"
CURVE_CSV="$(abs_path "$CURVE_CSV")"
RATE_CONFIG_CSV="$(abs_path "$RATE_CONFIG_CSV")"

mkdir -p "$OUT_DIR" "$UNICORN_MODEL_DIR" "$UNICORN_RESULTS_DIR" \
  "$UNICORN_TMP_DIR" "$UNICORN_BITSTREAM_DIR" "$UNICORN_DECODED_DIR" "${OUT_DIR}/logs"

require_dir "$OPENPCDET_TOOLS"
require_dir "$UNICORN_ROOT"

if [[ ! -d "$KITTI_VELODYNE" || ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  log "FOV-only KITTI data is missing; generating it first"
  OUTPUT_ROOT="$KITTI_ROOT" "$SCRIPT_DIR/prepare_kitti_fov.sh"
fi

require_dir "$KITTI_VELODYNE"
require_file "$TRAIN_SPLIT_FILE"
require_file "$SPLIT_FILE"

log "Using Unicorn test rate points: $RATES"
"$PYTHON_BIN" - "$RATES" "$RATE_CONFIG_CSV" <<'PY'
import csv
import sys
from pathlib import Path

rates = []
for rate_id, item in enumerate(sys.argv[1].split(',')):
    item = item.strip()
    if not item:
        continue
    parts = item.split(':')
    if len(parts) != 3:
        raise SystemExit(f'Bad rate point {item!r}; expected scale_AE:scale_SR:posQuantscale')
    scale_ae, scale_sr, posqscale = parts
    rates.append({
        'rate_id': rate_id,
        'rate_label': item,
        'scale_AE': int(scale_ae),
        'scale_SR': int(scale_sr),
        'posQuantscale': float(posqscale),
    })
if not rates:
    raise SystemExit('No Unicorn test rate points configured.')
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['rate_id', 'rate_label', 'scale_AE', 'scale_SR', 'posQuantscale'])
    writer.writeheader()
    writer.writerows(rates)
print(f'Wrote {len(rates)} Unicorn rate points -> {out}')
PY

if [[ "$RUN_LOSSLESS" == "1" ]]; then
  log "Step 1/5: train Unicorn lossless geometry module directly from KITTI .bin"
  cd "$SCRIPT_DIR"
  "$UNICORN_PYTHON_BIN" unicorn/train_unicorn_kitti.py \
    --unicorn_root "$UNICORN_ROOT" \
    --module lossless \
    --velodyne "$KITTI_VELODYNE" \
    --split_file "$TRAIN_SPLIT_FILE" \
    --model_save_folder "${UNICORN_MODEL_DIR}/lossless" \
    --train_posq "$TRAIN_POSQ" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --train_num "$TRAIN_NUM" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --learning_rate "$TRAIN_LR" \
    --lr_min "$TRAIN_LR" \
    --kernel_size "$KERNEL_SIZE" \
    --channels "$CHANNELS" \
    --block_layers "$BLOCK_LAYERS" \
    --block_type "$BLOCK_TYPE" \
    --stage "$LOSSLESS_STAGE" \
    --scale "$LOSSLESS_SCALE" \
    --weight_distortion 1 \
    --weight_bitrate 0 \
    --log_every "$TRAIN_LOG_EVERY" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    2>&1 | tee "${OUT_DIR}/logs/train_unicorn_lossless.log"
else
  log "Step 1/5 skipped: RUN_LOSSLESS=$RUN_LOSSLESS"
fi

if [[ "$RUN_SR" == "1" ]]; then
  log "Step 2/5: train Unicorn lossy SR module directly from KITTI .bin"
  cd "$SCRIPT_DIR"
  "$UNICORN_PYTHON_BIN" unicorn/train_unicorn_kitti.py \
    --unicorn_root "$UNICORN_ROOT" \
    --module sr \
    --velodyne "$KITTI_VELODYNE" \
    --split_file "$TRAIN_SPLIT_FILE" \
    --model_save_folder "${UNICORN_MODEL_DIR}/sr" \
    --train_posq "$TRAIN_POSQ" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --train_num "$TRAIN_NUM" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --learning_rate "$TRAIN_LR" \
    --lr_min "$TRAIN_LR" \
    --kernel_size "$KERNEL_SIZE" \
    --channels "$CHANNELS" \
    --block_layers "$BLOCK_LAYERS" \
    --block_type "$BLOCK_TYPE" \
    --log_every "$TRAIN_LOG_EVERY" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --sr_posQuantscaleList $POSQUANTSCALE_LIST \
    2>&1 | tee "${OUT_DIR}/logs/train_unicorn_sr.log"
else
  log "Step 2/5 skipped: RUN_SR=$RUN_SR"
fi

if [[ "$RUN_OFFSET" == "1" ]]; then
  log "Step 3/5: train Unicorn offset module directly from KITTI .bin"
  cd "$SCRIPT_DIR"
  "$UNICORN_PYTHON_BIN" unicorn/train_unicorn_kitti.py \
    --unicorn_root "$UNICORN_ROOT" \
    --module offset \
    --velodyne "$KITTI_VELODYNE" \
    --split_file "$TRAIN_SPLIT_FILE" \
    --model_save_folder "${UNICORN_MODEL_DIR}/offset" \
    --train_posq "$TRAIN_POSQ" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --train_num "$TRAIN_NUM" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --learning_rate "$TRAIN_LR" \
    --lr_min "$TRAIN_LR" \
    --offset_channels "$OFFSET_CHANNELS" \
    --kernel_size "$KERNEL_SIZE" \
    --block_layers "$BLOCK_LAYERS" \
    --log_every "$TRAIN_LOG_EVERY" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --posQuantscaleList $POSQUANTSCALE_LIST \
    2>&1 | tee "${OUT_DIR}/logs/train_unicorn_offset.log"
else
  log "Step 3/5 skipped: RUN_OFFSET=$RUN_OFFSET"
fi

if [[ "$RUN_UNICORN" == "1" ]]; then
  require_file "$LOSSLESS_LOW_CKPT"
  require_file "$SR_CKPT"
  require_file "$OFFSET_CKPT"
  log "Step 4/5: Unicorn rate/time/D1/D2 PSNR and decoded bins directly from KITTI .bin"
  UNICORN_EXTRA_ARGS=()
  if [[ "$UNICORN_RESUME" == "1" ]]; then
    UNICORN_EXTRA_ARGS+=(--resume)
  fi
  if [[ "$GPU_GUARD" == "1" && -n "$GPU_GUARD_ID" ]]; then
    UNICORN_EXTRA_ARGS+=(--gpu_guard_id "$GPU_GUARD_ID" --gpu_guard_interval "$GPU_GUARD_INTERVAL")
    log "GPU guard enabled for nvidia-smi GPU id: $GPU_GUARD_ID"
  else
    UNICORN_EXTRA_ARGS+=(--disable_gpu_guard)
    log "GPU guard disabled"
  fi
  cd "$SCRIPT_DIR"
  "$UNICORN_PYTHON_BIN" unicorn/unicorn_rates_direct.py \
    --unicorn_root "$UNICORN_ROOT" \
    --testdata "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --train_posq "$TRAIN_POSQ" \
    --results "$UNICORN_RESULTS_DIR" \
    --tmp_dir "$UNICORN_TMP_DIR" \
    --bitstream_dir "$UNICORN_BITSTREAM_DIR" \
    --decoded_dir "$UNICORN_DECODED_DIR" \
    --rates "$RATES" \
    --rate_config_csv "$RATE_CONFIG_CSV" \
    --ckptdir_low "$LOSSLESS_LOW_CKPT" \
    --ckptdir_sr_low "$SR_CKPT" \
    --ckptdir_offset "$OFFSET_CKPT" \
    --channels "$CHANNELS" \
    --offset_channels "$OFFSET_CHANNELS" \
    --kernel_size "$KERNEL_SIZE" \
    --block_layers "$BLOCK_LAYERS" \
    --block_type "$BLOCK_TYPE" \
    --resolution "$RESOLUTION" \
    "${UNICORN_EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/logs/unicorn_rates.log"
else
  log "Step 4/5 skipped: RUN_UNICORN=0"
fi

RATE_IDS="$("$PYTHON_BIN" - "$RATE_CONFIG_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline='') as f:
    print(','.join(row['rate_id'] for row in csv.DictReader(f)))
PY
)"

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 5/5: evaluate Unicorn decoded AP with OpenPCDet"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$DET_CKPT"
  "$PYTHON_BIN" "${SCRIPT_DIR}/unicorn/test_unicorn_decoded.py" \
    --cfg_file "$CFG_FILE" \
    --ckpt "$DET_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --extra_tag unicorn_decoded \
    --eval_tag default \
    --decoded_dir "$UNICORN_DECODED_DIR" \
    --rate_ids "$RATE_IDS" \
    2>&1 | tee "${OUT_DIR}/logs/ap_unicorn_all_rates.log"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" unicorn/parse_unicorn_ap_logs.py \
    --combined_log "${OUT_DIR}/logs/ap_unicorn_all_rates.log" \
    --out "$AP_CSV"
else
  log "Step 5/5 skipped: RUN_AP=0"
fi

if [[ -f "${UNICORN_RESULTS_DIR}/unicorn_average.csv" && -f "$AP_CSV" ]]; then
  log "Merge Unicorn curve CSV"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" unicorn/merge_unicorn_curve.py \
    --rate_csv "${UNICORN_RESULTS_DIR}/unicorn_average.csv" \
    --ap_csv "$AP_CSV" \
    --out "$CURVE_CSV"
else
  log "Skip curve merge: rate or AP CSV is not available in this partial run"
fi

if [[ "$CLEAN_INTERMEDIATE" == "1" ]]; then
  log "Clean Unicorn intermediate files"
  rm -rf "$UNICORN_TMP_DIR" "$UNICORN_BITSTREAM_DIR" "$UNICORN_DECODED_DIR"
else
  log "Intermediate files kept: CLEAN_INTERMEDIATE=$CLEAN_INTERMEDIATE"
fi

log "Done"
log "Unicorn lossless checkpoint: $LOSSLESS_LOW_CKPT"
log "Unicorn SR checkpoint: $SR_CKPT"
log "Unicorn offset checkpoint: $OFFSET_CKPT"
log "Unicorn average CSV: ${UNICORN_RESULTS_DIR}/unicorn_average.csv"
log "Unicorn AP CSV: $AP_CSV"
log "Unicorn curve CSV: $CURVE_CSV"
log "Unicorn rate config CSV: $RATE_CONFIG_CSV"
