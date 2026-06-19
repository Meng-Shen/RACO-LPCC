#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPCDET_TOOLS="${SCRIPT_DIR}/OpenPCDet/tools"

DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_fov_geometry.yaml}"
CKPT="${CKPT:-ckpt/model_non_reflectance.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
SCALES="${SCALES:-1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"
GPCC_CFG="${GPCC_CFG:-${SCRIPT_DIR}/extention/kitti.cfg}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/baseline_fov}"
GPCC_RESULTS_DIR="${GPCC_RESULTS_DIR:-${OUT_DIR}/gpcc}"
GPCC_TMP_DIR="${GPCC_TMP_DIR:-${OUT_DIR}/tmp_gpcc}"
AP_CSV="${AP_CSV:-${OUT_DIR}/baseline_ap.csv}"
CURVE_CSV="${CURVE_CSV:-${OUT_DIR}/baseline_gpcc_curve.csv}"

RUN_AP="${RUN_AP:-1}"
RUN_GPCC="${RUN_GPCC:-1}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$SCRIPT_DIR" "$1" ;;
  esac
}

OUT_DIR="$(abs_path "$OUT_DIR")"
GPCC_RESULTS_DIR="$(abs_path "$GPCC_RESULTS_DIR")"
GPCC_TMP_DIR="$(abs_path "$GPCC_TMP_DIR")"
AP_CSV="$(abs_path "$AP_CSV")"
CURVE_CSV="$(abs_path "$CURVE_CSV")"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

mkdir -p "$OUT_DIR" "$GPCC_RESULTS_DIR" "$GPCC_TMP_DIR"
require_dir "$OPENPCDET_TOOLS"

if [[ ! -d "$KITTI_VELODYNE" ||
      ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  log "FOV-only KITTI data is missing; generating it first"
  OUTPUT_ROOT="$KITTI_ROOT" "$SCRIPT_DIR/prepare_kitti_fov.sh"
fi

require_dir "$KITTI_VELODYNE"
require_file "$SPLIT_FILE"
require_file "$GPCC_CFG"

AP_LOG="${AP_LOG:-}"

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 1/3: evaluate baseline AP for global quantization scales"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$CKPT"
  before_file="${OUT_DIR}/ap_logs_before.txt"
  after_file="${OUT_DIR}/ap_logs_after.txt"
  find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_pos_*.txt' 2>/dev/null | sort > "$before_file" || true
  "$PYTHON_BIN" test_pos.py \
    --cfg_file "$CFG_FILE" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --scales "$SCALES"
  find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_pos_*.txt' 2>/dev/null | sort > "$after_file" || true
  AP_LOG="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
  if [[ -z "$AP_LOG" ]]; then
    AP_LOG="$(find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_pos_*.txt' 2>/dev/null | sort | tail -n 1)"
  fi
  require_file "$AP_LOG"
  log "Parsing AP log: $AP_LOG"
  "$PYTHON_BIN" parse_baseline_ap_log.py --log "$AP_LOG" --out "$AP_CSV"
else
  log "Step 1/3 skipped: RUN_AP=0"
  require_file "$AP_CSV"
fi

if [[ "$RUN_GPCC" == "1" ]]; then
  log "Step 2/3: measure baseline G-PCC bpp/enc_time/dec_time"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" GPCC/baseline_rates.py \
    --testdata "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --scales "$SCALES" \
    --results "$GPCC_RESULTS_DIR" \
    --tmp_dir "$GPCC_TMP_DIR" \
    --cfg "$GPCC_CFG"
else
  log "Step 2/3 skipped: RUN_GPCC=0"
  require_file "${GPCC_RESULTS_DIR}/gpcc_baseline_average.csv"
fi

log "Step 3/3: merge AP and G-PCC metrics into curve CSV"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" merge_baseline_curve.py \
  --ap_csv "$AP_CSV" \
  --gpcc_csv "${GPCC_RESULTS_DIR}/gpcc_baseline_average.csv" \
  --out "$CURVE_CSV"

log "Done"
log "AP CSV: $AP_CSV"
log "G-PCC CSV: ${GPCC_RESULTS_DIR}/gpcc_baseline_average.csv"
log "Curve CSV: $CURVE_CSV"
