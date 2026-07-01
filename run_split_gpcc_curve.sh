#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPCDET_TOOLS="${SCRIPT_DIR}/OpenPCDet/tools"
MMDET_DIR="${SCRIPT_DIR}/mmdetection3d"
export PYTHONPATH="${MMDET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_fov_geometry.yaml}"
CKPT="${CKPT:-ckpt/model_non_reflectance.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
SPLIT_SCALES="${SPLIT_SCALES:-1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048}"

SEG_CFG="${SEG_CFG:-${MMDET_DIR}/configs/minkunet/minkunet34_w32_minkowski_geometry_kitti_box_seg.py}"
SEG_CKPT="${SEG_CKPT:-${MMDET_DIR}/work_dirs/minkunet_kitti_fov_box_seg_geometry/best_foreground_epoch_33.pth}"
SEG_FG_THRESHOLD="${SEG_FG_THRESHOLD:-0.35}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"
GPCC_CFG="${GPCC_CFG:-${SCRIPT_DIR}/extention/kitti.cfg}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/split_gpcc_fov}"
MASK_DIR="${MASK_DIR:-${OUT_DIR}/seg_masks}"
SEG_TIME_CSV="${SEG_TIME_CSV:-${OUT_DIR}/seg_time.csv}"
GPCC_RESULTS_DIR="${GPCC_RESULTS_DIR:-${OUT_DIR}/gpcc}"
GPCC_TMP_DIR="${GPCC_TMP_DIR:-${OUT_DIR}/tmp_gpcc}"
AP_CSV="${AP_CSV:-${OUT_DIR}/split_ap.csv}"
CURVE_CSV="${CURVE_CSV:-${OUT_DIR}/split_gpcc_curve.csv}"

RUN_SEG="${RUN_SEG:-1}"
RUN_AP="${RUN_AP:-1}"
RUN_GPCC="${RUN_GPCC:-1}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$SCRIPT_DIR" "$1" ;;
  esac
}

OUT_DIR="$(abs_path "$OUT_DIR")"
MASK_DIR="$(abs_path "$MASK_DIR")"
SEG_TIME_CSV="$(abs_path "$SEG_TIME_CSV")"
GPCC_RESULTS_DIR="$(abs_path "$GPCC_RESULTS_DIR")"
GPCC_TMP_DIR="$(abs_path "$GPCC_TMP_DIR")"
AP_CSV="$(abs_path "$AP_CSV")"
CURVE_CSV="$(abs_path "$CURVE_CSV")"
SEG_CFG="$(abs_path "$SEG_CFG")"
SEG_CKPT="$(abs_path "$SEG_CKPT")"
KITTI_ROOT="$(abs_path "$KITTI_ROOT")"
KITTI_VELODYNE="$(abs_path "$KITTI_VELODYNE")"
SPLIT_FILE="$(abs_path "$SPLIT_FILE")"
GPCC_CFG="$(abs_path "$GPCC_CFG")"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

mkdir -p "$OUT_DIR" "$MASK_DIR" "$GPCC_RESULTS_DIR" "$GPCC_TMP_DIR"
require_dir "$OPENPCDET_TOOLS"

if [[ ! -d "$KITTI_VELODYNE" ||
      ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  log "FOV-only KITTI data is missing; generating it first"
  OUTPUT_ROOT="$KITTI_ROOT" "$SCRIPT_DIR/prepare_kitti_fov.sh"
fi

require_dir "$KITTI_VELODYNE"
require_file "$SPLIT_FILE"
require_file "$GPCC_CFG"

if [[ "$RUN_SEG" == "1" ]]; then
  log "Step 1/4: generate or reuse foreground/background masks"
  require_file "$SEG_CFG"
  require_file "$SEG_CKPT"
  cd "$OPENPCDET_TOOLS"
  "$PYTHON_BIN" generate_masks.py \
    --val_txt "$SPLIT_FILE" \
    --bin_dir "$KITTI_VELODYNE" \
    --out_dir "$MASK_DIR" \
    --seg_cfg_file "$SEG_CFG" \
    --seg_ckpt "$SEG_CKPT" \
    --time_csv "$SEG_TIME_CSV" \
    --fg_threshold "$SEG_FG_THRESHOLD" \
    --device "cuda:0"
else
  log "Step 1/4 skipped: RUN_SEG=0"
fi

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 2/4: evaluate Split-GPCC AP for foreground/background quantization pairs"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$CKPT"
  before_file="${OUT_DIR}/ap_logs_before.txt"
  after_file="${OUT_DIR}/ap_logs_after.txt"
  find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_split_*.txt' 2>/dev/null | sort > "$before_file" || true
  "$PYTHON_BIN" test_split.py \
    --cfg_file "$CFG_FILE" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --mask_dir "$MASK_DIR" \
    --quant_map "$SPLIT_SCALES"
  find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_split_*.txt' 2>/dev/null | sort > "$after_file" || true
  AP_LOG="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
  if [[ -z "$AP_LOG" ]]; then
    AP_LOG="$(find "${SCRIPT_DIR}/OpenPCDet/output" -name 'log_eval_split_*.txt' 2>/dev/null | sort | tail -n 1)"
  fi
  require_file "$AP_LOG"
  log "Parsing split AP log: $AP_LOG"
  "$PYTHON_BIN" parse_split_ap_log.py --log "$AP_LOG" --out "$AP_CSV"
else
  log "Step 2/4 skipped: RUN_AP=0"
  require_file "$AP_CSV"
fi

if [[ "$RUN_GPCC" == "1" ]]; then
  log "Step 3/4: measure Split-GPCC bpp and time from cached masks"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" GPCC/split_rates.py \
    --testdata "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --mask_dir "$MASK_DIR" \
    --seg_time_csv "$SEG_TIME_CSV" \
    --quant_map "$SPLIT_SCALES" \
    --results "$GPCC_RESULTS_DIR" \
    --tmp_dir "$GPCC_TMP_DIR" \
    --cfg "$GPCC_CFG"
else
  log "Step 3/4 skipped: RUN_GPCC=0"
  require_file "${GPCC_RESULTS_DIR}/split_average_results.csv"
fi

log "Step 4/4: merge AP and Split-GPCC metrics into curve CSV"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" merge_split_curve.py \
  --ap_csv "$AP_CSV" \
  --gpcc_csv "${GPCC_RESULTS_DIR}/split_average_results.csv" \
  --out "$CURVE_CSV"

log "Done"
log "AP CSV: $AP_CSV"
log "Mask dir: $MASK_DIR"
log "Seg time CSV: $SEG_TIME_CSV"
log "Split-GPCC CSV: ${GPCC_RESULTS_DIR}/split_average_results.csv"
log "Curve CSV: $CURVE_CSV"
