#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/OpenPCDet/tools"

PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_fov_geometry.yaml}"
CKPT="${CKPT:-ckpt/latest_model.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"

SPLIT_FILE="${SPLIT_FILE:-../data/kitti_fov/ImageSets/val.txt}"
EVAL_DIR="${EVAL_DIR:-../output/kitti_models/pv_rcnn/default/eval/epoch_no_number/val/default}"
MASK_DIR="${MASK_DIR:-../output/eval/test_seg_masks}"

# 点云量化步长组合只在这里定义一次。
# 格式："前景步长,背景步长;前景步长,背景步长;..."
# 顺序就是 test_split.py 生成 combo_0、combo_1、... 的顺序。
# 约定：最后一个 combo 是最高码率/基准，对应 JUCP Label 0；倒数第二个对应 Label 1；依此类推。
QUANT_MAP="${QUANT_MAP:-1/256,1/1024;2/256,1/1024;3/256,1/1024;1/64,1/1024;1/64,1/512;1/64,1.25/512;1/64,1.5/512}"

# 三个目标类别各自的 AP 下降阈值。
# 这三个值会同时用于：
# 1. 自动生成 JUCP_CSV 文件名；
# 2. 传入 jucp_split.py 决定每帧选择哪个压缩 Label。
JUCP_CAR_THRESHOLD="${JUCP_CAR_THRESHOLD:-0}"
JUCP_PED_THRESHOLD="${JUCP_PED_THRESHOLD:-0}"
JUCP_CYC_THRESHOLD="${JUCP_CYC_THRESHOLD:-0}"

AP_CSV="${AP_CSV:-split_AP.csv}"
JUCP_CSV="${JUCP_CSV:-jucp${JUCP_CAR_THRESHOLD}_${JUCP_PED_THRESHOLD}_${JUCP_CYC_THRESHOLD}.csv}"

TEST_SPLIT_WORKERS="${TEST_SPLIT_WORKERS:-4}"
NEW_SPLIT_WORKERS="${NEW_SPLIT_WORKERS:-64}"
JUCP_EVAL_WORKERS="${JUCP_EVAL_WORKERS:-4}"

RUN_TEST_SPLIT="${RUN_TEST_SPLIT:-1}"
RUN_NEW_SPLIT="${RUN_NEW_SPLIT:-1}"
RUN_JUCP_SPLIT="${RUN_JUCP_SPLIT:-1}"
RUN_TEST_JUCP_SPLIT="${RUN_TEST_JUCP_SPLIT:-1}"

usage() {
    cat <<'EOF'
Usage:
  ./juqp.sh

Common overrides:
  CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml
  CKPT=ckpt/latest_model.pth
  BATCH_SIZE=8
  MASK_DIR=../output/eval/seg_masks
  SPLIT_FILE=../data/kitti/ImageSets/val.txt
  EVAL_DIR=../output/kitti_models/pv_rcnn/default/eval/epoch_no_number/val/default
  AP_CSV=split_AP.csv

Quantization map, define once here and pass to all Python scripts:
  QUANT_MAP='1/256,0;2/256,0;3/256,0;1/64,0;1/64,1/512;1/64,1.25/512;1/64,1.5/512'

JUCP thresholds, used both for CSV filename and jucp_split.py:
  JUCP_CAR_THRESHOLD=0.0045
  JUCP_PED_THRESHOLD=0.05
  JUCP_CYC_THRESHOLD=0.075
  JUCP_CSV=jucp0.0045_0.05_0.075.csv   # optional; default is generated automatically

Workers:
  TEST_SPLIT_WORKERS=4
  NEW_SPLIT_WORKERS=64
  JUCP_EVAL_WORKERS=4

Step switches, set to 0 to skip:
  RUN_TEST_SPLIT=1
  RUN_NEW_SPLIT=1
  RUN_JUCP_SPLIT=1
  RUN_TEST_JUCP_SPLIT=1

Examples:
  CUDA_VISIBLE_DEVICES=0 NEW_SPLIT_WORKERS=16 ./juqp.sh

  JUCP_CAR_THRESHOLD=0.003 \
  JUCP_PED_THRESHOLD=0.04 \
  JUCP_CYC_THRESHOLD=0.06 \
  ./juqp.sh

  QUANT_MAP='1/512,0;1/256,0;1/64,1.5/512' ./juqp.sh

  RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp.sh
EOF
}

log() {
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local desc="$2"
    [[ -f "$path" ]] || die "Missing ${desc}: ${path}"
}

require_dir() {
    local path="$1"
    local desc="$2"
    [[ -d "$path" ]] || die "Missing ${desc}: ${path}"
}

run_cmd() {
    log "Running: $*"
    "$@"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_dir "$TOOLS_DIR" "OpenPCDet tools directory"
cd "$TOOLS_DIR"

require_file "$CFG_FILE" "OpenPCDet config"
require_file "$CKPT" "checkpoint"

if [[ "$RUN_TEST_SPLIT" == "1" || "$RUN_TEST_JUCP_SPLIT" == "1" ]]; then
    require_dir "$MASK_DIR" "semantic segmentation mask directory"
fi

if [[ "$RUN_NEW_SPLIT" == "1" ]]; then
    require_file "$SPLIT_FILE" "split file"
fi

log "JUQP pipeline started"
log "Tools directory: $TOOLS_DIR"
log "Config: $CFG_FILE"
log "Checkpoint: $CKPT"
log "Quant map: $QUANT_MAP"
log "JUCP thresholds: Car=${JUCP_CAR_THRESHOLD}, Pedestrian=${JUCP_PED_THRESHOLD}, Cyclist=${JUCP_CYC_THRESHOLD}"
log "AP CSV: $AP_CSV"
log "JUCP CSV: $JUCP_CSV"

if [[ "$RUN_TEST_SPLIT" == "1" ]]; then
    log "Step 1/4: generate split-quantization prediction results"
    run_cmd "$PYTHON_BIN" test_split.py \
        --cfg_file "$CFG_FILE" \
        --ckpt "$CKPT" \
        --batch_size "$BATCH_SIZE" \
        --workers "$TEST_SPLIT_WORKERS" \
        --mask_dir "$MASK_DIR" \
        --quant_map "$QUANT_MAP"
else
    log "Step 1/4 skipped: RUN_TEST_SPLIT=0"
fi

if [[ "$RUN_NEW_SPLIT" == "1" ]]; then
    log "Step 2/4: calculate per-frame AP sensitivity matrix"
    require_dir "$EVAL_DIR" "test_split eval output directory"
    run_cmd "$PYTHON_BIN" new_split.py \
        --cfg_file "$CFG_FILE" \
        --split_file "$SPLIT_FILE" \
        --eval_dir "$EVAL_DIR" \
        --out_csv "$AP_CSV" \
        --workers "$NEW_SPLIT_WORKERS" \
        --quant_map "$QUANT_MAP"
    require_file "$AP_CSV" "AP CSV output"
else
    log "Step 2/4 skipped: RUN_NEW_SPLIT=0"
    require_file "$AP_CSV" "existing AP CSV"
fi

if [[ "$RUN_JUCP_SPLIT" == "1" ]]; then
    log "Step 3/4: derive JUCP labels from AP matrix"
    run_cmd "$PYTHON_BIN" jucp_split.py \
        --ap_csv "$AP_CSV" \
        --out_csv "$JUCP_CSV" \
        --car_threshold "$JUCP_CAR_THRESHOLD" \
        --ped_threshold "$JUCP_PED_THRESHOLD" \
        --cyc_threshold "$JUCP_CYC_THRESHOLD"
    require_file "$JUCP_CSV" "JUCP CSV output"
else
    log "Step 3/4 skipped: RUN_JUCP_SPLIT=0"
    require_file "$JUCP_CSV" "existing JUCP CSV"
fi

if [[ "$RUN_TEST_JUCP_SPLIT" == "1" ]]; then
    log "Step 4/4: evaluate JUCP split compression"
    run_cmd "$PYTHON_BIN" test_jucp_split.py \
        --cfg_file "$CFG_FILE" \
        --batch_size "$BATCH_SIZE" \
        --ckpt "$CKPT" \
        --jucp_csv "$JUCP_CSV" \
        --mask_dir "$MASK_DIR" \
        --workers "$JUCP_EVAL_WORKERS" \
        --quant_map "$QUANT_MAP"
else
    log "Step 4/4 skipped: RUN_TEST_JUCP_SPLIT=0"
fi

log "JUQP pipeline finished"
log "AP CSV: ${TOOLS_DIR}/${AP_CSV}"
log "JUCP CSV: ${TOOLS_DIR}/${JUCP_CSV}"
