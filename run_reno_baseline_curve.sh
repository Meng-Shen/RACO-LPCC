#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPCDET_TOOLS="${SCRIPT_DIR}/OpenPCDet/tools"

DEFAULT_OPC_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
DEFAULT_RENO_PYTHON="/home/sm/miniconda3/envs/reno/bin/python"
if [[ ! -x "$DEFAULT_RENO_PYTHON" ]]; then
  DEFAULT_RENO_PYTHON="$DEFAULT_OPC_PYTHON"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_OPC_PYTHON}"
RENO_PYTHON_BIN="${RENO_PYTHON_BIN:-$DEFAULT_RENO_PYTHON}"

RENO_ROOT="${RENO_ROOT:-/public/DATA/sm/RENO}"
CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_fov_geometry.yaml}"
DET_CKPT="${DET_CKPT:-ckpt/model_non_reflectance.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
SCALES="${SCALES:-1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512,1/2048}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-${KITTI_ROOT}/ImageSets/train.txt}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/reno_fov}"
RENO_MODEL_DIR="${RENO_MODEL_DIR:-${OUT_DIR}/model}"
RENO_CKPT="${RENO_CKPT:-${RENO_MODEL_DIR}/ckpt.pt}"
RENO_RESULTS_DIR="${RENO_RESULTS_DIR:-${OUT_DIR}/reno}"
RENO_TMP_DIR="${RENO_TMP_DIR:-${OUT_DIR}/tmp}"
RENO_BITSTREAM_DIR="${RENO_BITSTREAM_DIR:-${OUT_DIR}/bitstreams}"
AP_CSV="${AP_CSV:-${OUT_DIR}/reno_ap.csv}"
CURVE_CSV="${CURVE_CSV:-${OUT_DIR}/reno_full_curve.csv}"

RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_RENO="${RUN_RENO:-1}"
RUN_AP="${RUN_AP:-1}"
MAX_STEPS="${MAX_STEPS:-170000}"
RESOLUTION="${RESOLUTION:-59.70}"
TRAIN_POSQ="${TRAIN_POSQ:-4.0}"

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
RENO_MODEL_DIR="$(abs_path "$RENO_MODEL_DIR")"
RENO_RESULTS_DIR="$(abs_path "$RENO_RESULTS_DIR")"
RENO_TMP_DIR="$(abs_path "$RENO_TMP_DIR")"
RENO_BITSTREAM_DIR="$(abs_path "$RENO_BITSTREAM_DIR")"
AP_CSV="$(abs_path "$AP_CSV")"
CURVE_CSV="$(abs_path "$CURVE_CSV")"

mkdir -p "$OUT_DIR" "$RENO_RESULTS_DIR" "$RENO_TMP_DIR" "$RENO_BITSTREAM_DIR" "${OUT_DIR}/logs"
require_dir "$OPENPCDET_TOOLS"
require_dir "$RENO_ROOT"

if [[ ! -d "$KITTI_VELODYNE" || ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  log "FOV-only KITTI data is missing; generating it first"
  OUTPUT_ROOT="$KITTI_ROOT" "$SCRIPT_DIR/prepare_kitti_fov.sh"
fi

require_dir "$KITTI_VELODYNE"
require_file "$SPLIT_FILE"

if [[ "$RUN_TRAIN" == "1" ]]; then
  log "Step 1/4: train RENO on KITTI"
  mkdir -p "$RENO_MODEL_DIR"
  cd "$SCRIPT_DIR"
  "$RENO_PYTHON_BIN" reno/train_kitti.py \
    --reno_root "$RENO_ROOT" \
    --training_data "${KITTI_VELODYNE}/*.bin" \
    --model_save_folder "$RENO_MODEL_DIR" \
    --valid_samples "$TRAIN_SPLIT_FILE" \
    --batch_size 1 \
    --train_posq "$TRAIN_POSQ" \
    --learning_rate 0.0005 \
    --max_steps "$MAX_STEPS" \
    2>&1 | tee "${OUT_DIR}/logs/train_reno.log"
else
  log "Step 1/4 skipped: RUN_TRAIN=0"
fi
require_file "$RENO_CKPT"

if [[ "$RUN_RENO" == "1" ]]; then
  log "Step 2/4: RENO encode/decode, rate/time, D1/D2 PSNR"
  cd "$SCRIPT_DIR"
  "$RENO_PYTHON_BIN" reno/reno_rates.py \
    --reno_root "$RENO_ROOT" \
    --testdata "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --scales "$SCALES" \
    --ckpt "$RENO_CKPT" \
    --results "$RENO_RESULTS_DIR" \
    --tmp_dir "$RENO_TMP_DIR" \
    --bitstream_dir "$RENO_BITSTREAM_DIR" \
    --kitti_root "$KITTI_ROOT" \
    --resolution "$RESOLUTION" \
    2>&1 | tee "${OUT_DIR}/logs/reno_rates.log"
else
  log "Step 2/4 skipped: RUN_RENO=0"
  require_file "${RENO_RESULTS_DIR}/reno_average.csv"
fi

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 3/4: evaluate RENO AP with OpenPCDet"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$DET_CKPT"
  "$PYTHON_BIN" "${SCRIPT_DIR}/reno/test_reno_pos.py" \
    --cfg_file "$CFG_FILE" \
    --ckpt "$DET_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --extra_tag reno_quantized \
    --eval_tag default \
    --scales "$SCALES" \
    2>&1 | tee "${OUT_DIR}/logs/ap_reno_all_rates.log"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" reno/parse_reno_ap_logs.py \
    --log_dir "${OUT_DIR}/logs" \
    --combined_log "${OUT_DIR}/logs/ap_reno_all_rates.log" \
    --out "$AP_CSV"
else
  log "Step 3/4 skipped: RUN_AP=0"
  require_file "$AP_CSV"
fi

log "Step 4/4: merge RENO curve CSV"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" reno/merge_reno_curve.py \
  --rate_csv "${RENO_RESULTS_DIR}/reno_average.csv" \
  --ap_csv "$AP_CSV" \
  --out "$CURVE_CSV"

log "Done"
log "RENO average CSV: ${RENO_RESULTS_DIR}/reno_average.csv"
log "RENO AP CSV: $AP_CSV"
log "RENO curve CSV: $CURVE_CSV"
