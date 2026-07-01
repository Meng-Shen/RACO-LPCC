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
DET_CKPT="${DET_CKPT:-ckpt/model_w_juqp.pth}"
DET_EXTRA_TAG="${DET_EXTRA_TAG:-juqp_model_w_juqp_val}"
DET_EVAL_TAG="${DET_EVAL_TAG:-default}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"

QUANT_MAP="${QUANT_MAP:-1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048}"
THRESHOLDS="${THRESHOLDS:-0,0,0;0.0005,0.01,0.02;0.00075,0.02,0.035;0.00125,0.03,0.045;0.00175,0.04,0.06;0.00225,0.05,0.075}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"
MASK_DIR="${MASK_DIR:-${SCRIPT_DIR}/point_pairs/split_gpcc_fov/seg_masks}"
SEG_TIME_CSV="${SEG_TIME_CSV:-${SCRIPT_DIR}/point_pairs/split_gpcc_fov/seg_time.csv}"
CFG_STEM="$(basename "${CFG_FILE%.yaml}")"
SPLIT_EVAL_DIR="${SPLIT_EVAL_DIR:-${SCRIPT_DIR}/OpenPCDet/output/kitti_models/${CFG_STEM}/${DET_EXTRA_TAG}/eval/epoch_no_number/val/${DET_EVAL_TAG}}"
SPLIT_DETAILS_CSV="${SPLIT_DETAILS_CSV:-${SCRIPT_DIR}/point_pairs/split_gpcc_fov/gpcc/split_all_details.csv}"

ROUTER_CKPT="${ROUTER_CKPT:-${OPENPCDET_TOOLS}/router_work_dirs/cost_proxy_model_w_juqp/best.pth}"
ROUTER_CALIBRATION="${ROUTER_CALIBRATION:-${OPENPCDET_TOOLS}/router_work_dirs/cost_proxy_model_w_juqp/calibration.pth}"
ROUTER_DEVICE="${ROUTER_DEVICE:-cuda}"
SELECTION_POLICY="${SELECTION_POLICY:-hard}"
BPP_ESTIMATE="${BPP_ESTIMATE:-mean}"
DEBT_TARGET="${DEBT_TARGET:-car}"
DEBT_ALPHA="${DEBT_ALPHA:-1.0}"
DEBT_BETA="${DEBT_BETA:-0.5}"
DEBT_MAX_EXTRA="${DEBT_MAX_EXTRA:-0.0015}"
DEBT_MIN_THRESHOLD_RATIO="${DEBT_MIN_THRESHOLD_RATIO:-0.5}"
DEBT_MIN_SAVING_PER_COST="${DEBT_MIN_SAVING_PER_COST:-0.0}"
LAGRANGE_LAMBDAS="${LAGRANGE_LAMBDAS:-0,0.0005,0.001,0.002,0.005,0.01,0.02}"
LAGRANGE_CLASS_WEIGHTS="${LAGRANGE_CLASS_WEIGHTS:-1,0.3,0.1}"
LAGRANGE_MAX_LABELS="${LAGRANGE_MAX_LABELS:-}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/router_gpcc_fov}"
LABEL_DIR="${LABEL_DIR:-${OUT_DIR}/labels}"
GPCC_RESULTS_DIR="${GPCC_RESULTS_DIR:-${OUT_DIR}/gpcc}"
AP_CSV="${AP_CSV:-${OUT_DIR}/router_ap.csv}"
CURVE_CSV="${CURVE_CSV:-${OUT_DIR}/router_gpcc_curve.csv}"

RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_TEST_SPLIT="${RUN_TEST_SPLIT:-1}"
RUN_AP="${RUN_AP:-1}"
RUN_GPCC="${RUN_GPCC:-1}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$SCRIPT_DIR" "$1" ;;
  esac
}

KITTI_ROOT="$(abs_path "$KITTI_ROOT")"
KITTI_VELODYNE="$(abs_path "$KITTI_VELODYNE")"
SPLIT_FILE="$(abs_path "$SPLIT_FILE")"
SPLIT_EVAL_DIR="$(abs_path "$SPLIT_EVAL_DIR")"
SPLIT_DETAILS_CSV="$(abs_path "$SPLIT_DETAILS_CSV")"
ROUTER_CKPT="$(abs_path "$ROUTER_CKPT")"
ROUTER_CALIBRATION="$(abs_path "$ROUTER_CALIBRATION")"
OUT_DIR="$(abs_path "$OUT_DIR")"
LABEL_DIR="$(abs_path "$LABEL_DIR")"
GPCC_RESULTS_DIR="$(abs_path "$GPCC_RESULTS_DIR")"
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

mkdir -p "$OUT_DIR" "$LABEL_DIR" "$GPCC_RESULTS_DIR"
require_dir "$OPENPCDET_TOOLS"
require_dir "$KITTI_VELODYNE"
require_file "$SPLIT_FILE"
require_file "$ROUTER_CKPT"

MANIFEST="${LABEL_DIR}/router_manifest.json"

if [[ "$RUN_EXPORT" == "1" ]]; then
  log "Step 1/4: export router AP-drop costs and per-threshold JUQP labels"
  cd "$OPENPCDET_TOOLS"
  "$PYTHON_BIN" export_router_jucp.py \
    --velodyne_dir "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --ckpt "$ROUTER_CKPT" \
    --calibration "$ROUTER_CALIBRATION" \
    --thresholds "$THRESHOLDS" \
    --quant_map "$QUANT_MAP" \
    --out_dir "$LABEL_DIR" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$ROUTER_DEVICE" \
    --selection_policy "$SELECTION_POLICY" \
    --split_details_csv "$SPLIT_DETAILS_CSV" \
    --bpp_estimate "$BPP_ESTIMATE" \
    --debt_target "$DEBT_TARGET" \
    --debt_alpha "$DEBT_ALPHA" \
    --debt_beta "$DEBT_BETA" \
    --debt_max_extra "$DEBT_MAX_EXTRA" \
    --debt_min_threshold_ratio "$DEBT_MIN_THRESHOLD_RATIO" \
    --debt_min_saving_per_cost "$DEBT_MIN_SAVING_PER_COST" \
    --lagrange_lambdas "$LAGRANGE_LAMBDAS" \
    --lagrange_class_weights "$LAGRANGE_CLASS_WEIGHTS" \
    --lagrange_max_labels "$LAGRANGE_MAX_LABELS"
else
  log "Step 1/4 skipped: RUN_EXPORT=0"
  require_file "$MANIFEST"
fi

if [[ "$RUN_TEST_SPLIT" == "1" ]]; then
  log "Step 2/5: run detector once per fixed quantization combo and save result.pkl files"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$DET_CKPT"
  require_dir "$MASK_DIR"
  "$PYTHON_BIN" test_split.py \
    --cfg_file "$CFG_FILE" \
    --ckpt "$DET_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --extra_tag "$DET_EXTRA_TAG" \
    --eval_tag "$DET_EVAL_TAG" \
    --mask_dir "$MASK_DIR" \
    --quant_map "$QUANT_MAP"
else
  log "Step 2/5 skipped: RUN_TEST_SPLIT=0"
fi

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 3/5: evaluate router AP by selecting existing combo result.pkl files"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_dir "$SPLIT_EVAL_DIR"
  "$PYTHON_BIN" eval_router_from_pkls.py \
    --cfg_file "$CFG_FILE" \
    --eval_dir "$SPLIT_EVAL_DIR" \
    --quant_map "$QUANT_MAP" \
    --manifest "$MANIFEST" \
    --out "$AP_CSV" \
    --save_mixed_pkls_dir "${OUT_DIR}/mixed_result_pkls"
else
  log "Step 3/5 skipped: RUN_AP=0"
  require_file "$AP_CSV"
fi

if [[ "$RUN_GPCC" == "1" ]]; then
  log "Step 4/5: aggregate bpp and time from existing Split-GPCC per-frame details"
  cd "$SCRIPT_DIR"
  require_file "$SPLIT_DETAILS_CSV"
  "$PYTHON_BIN" GPCC/aggregate_router_rates.py \
    --split_details_csv "$SPLIT_DETAILS_CSV" \
    --split_file "$SPLIT_FILE" \
    --manifest "$MANIFEST" \
    --out_dir "$GPCC_RESULTS_DIR"
else
  log "Step 4/5 skipped: RUN_GPCC=0"
  require_file "${GPCC_RESULTS_DIR}/router_average_results.csv"
fi

log "Step 5/5: merge AP and router-assisted compression metrics"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" merge_router_curve.py \
  --ap_csv "$AP_CSV" \
  --gpcc_csv "${GPCC_RESULTS_DIR}/router_average_results.csv" \
  --out "$CURVE_CSV"

log "Done"
log "Router labels: $LABEL_DIR"
log "Router costs: ${LABEL_DIR}/router_costs.csv"
log "AP CSV: $AP_CSV"
log "GPCC CSV: ${GPCC_RESULTS_DIR}/router_average_results.csv"
log "Curve CSV: $CURVE_CSV"
