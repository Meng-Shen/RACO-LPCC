#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/OpenPCDet/tools"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Detection/AP-matrix inputs. The config evaluates the original KITTI train split
# as the test split, rather than the train_5to1_* detector split.
CFG_FILE="${CFG_FILE:-cfgs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml}"
CKPT="${CKPT:-ckpt/model_w_juqp.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_SPLIT_WORKERS="${TEST_SPLIT_WORKERS:-4}"
NEW_SPLIT_WORKERS="${NEW_SPLIT_WORKERS:-64}"

SPLIT_FILE="${SPLIT_FILE:-../data/kitti_fov/ImageSets/train.txt}"
MASK_DIR="${MASK_DIR:-../output/eval/train_seg_masks}"
EVAL_EXTRA_TAG="${EVAL_EXTRA_TAG:-juqp_model_w_juqp_train}"
EVAL_TAG="${EVAL_TAG:-default}"
CFG_STEM="$(basename "${CFG_FILE%.yaml}")"
EVAL_DIR="${EVAL_DIR:-../output/kitti_models/${CFG_STEM}/${EVAL_EXTRA_TAG}/eval/epoch_no_number/train/${EVAL_TAG}}"

QUANT_MAP="${QUANT_MAP:-1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048}"
THRESHOLDS="${THRESHOLDS:-0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075}"

AP_CSV="${AP_CSV:-split_AP_train_model_w_juqp.csv}"
JUQP_LABEL_DIR="${JUQP_LABEL_DIR:-juqp_train_labels_model_w_juqp}"

# Router proxy training. The original train split is split once for proxy
# training and cost-head calibration.
ROUTER_OUT_DIR="${ROUTER_OUT_DIR:-router_work_dirs/cost_proxy_model_w_juqp}"
ROUTER_SPLIT_DIR="${ROUTER_SPLIT_DIR:-${ROUTER_OUT_DIR}/splits}"
ROUTER_TRAIN_SPLIT="${ROUTER_TRAIN_SPLIT:-${ROUTER_SPLIT_DIR}/train_from_original_train.txt}"
ROUTER_VAL_SPLIT="${ROUTER_VAL_SPLIT:-${ROUTER_SPLIT_DIR}/val_from_original_train.txt}"
ROUTER_VAL_RATIO="${ROUTER_VAL_RATIO:-0.1666666667}"
ROUTER_SPLIT_SEED="${ROUTER_SPLIT_SEED:-2026}"
ROUTER_SHUFFLE="${ROUTER_SHUFFLE:-0}"

VELODYNE_DIR="${VELODYNE_DIR:-../data/kitti/training/velodyne}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-120}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-8}"
ROUTER_WORKERS="${ROUTER_WORKERS:-4}"
ROUTER_DEVICE="${ROUTER_DEVICE:-cuda}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-20}"

RUN_TEST_SPLIT="${RUN_TEST_SPLIT:-1}"
RUN_NEW_SPLIT="${RUN_NEW_SPLIT:-1}"
RUN_JUQP_LABELS="${RUN_JUQP_LABELS:-1}"
RUN_ROUTER_SPLIT="${RUN_ROUTER_SPLIT:-1}"
RUN_ROUTER_TRAIN="${RUN_ROUTER_TRAIN:-1}"

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

sanitize_threshold_name() {
  printf '%s' "$1" | tr ',' '_' | tr '.' 'p' | tr -d ' '
}

make_router_split() {
  run_cmd "$PYTHON_BIN" - "$SPLIT_FILE" "$ROUTER_TRAIN_SPLIT" "$ROUTER_VAL_SPLIT" \
    "$ROUTER_VAL_RATIO" "$ROUTER_SPLIT_SEED" "$ROUTER_SHUFFLE" <<'PY'
import random
import sys
from pathlib import Path

src, train_out, val_out, val_ratio, seed, shuffle = sys.argv[1:]
val_ratio = float(val_ratio)
seed = int(seed)
shuffle = str(shuffle).strip() == "1"

ids = [line.strip() for line in Path(src).read_text().splitlines() if line.strip()]
if len(ids) < 2:
    raise SystemExit(f"Need at least 2 frame ids in {src}")
if not 0.0 < val_ratio < 1.0:
    raise SystemExit("-- val ratio must be in (0,1)")

items = list(ids)
if shuffle:
    random.Random(seed).shuffle(items)

val_count = max(1, min(len(items) - 1, round(len(items) * val_ratio)))
train_ids = items[:-val_count]
val_ids = items[-val_count:]

Path(train_out).parent.mkdir(parents=True, exist_ok=True)
Path(val_out).parent.mkdir(parents=True, exist_ok=True)
Path(train_out).write_text("".join(f"{x}\n" for x in train_ids))
Path(val_out).write_text("".join(f"{x}\n" for x in val_ids))
print(f"source={src} total={len(items)} train={len(train_ids)} val={len(val_ids)} shuffle={shuffle}")
print(f"train_split={train_out}")
print(f"val_split={val_out}")
PY
}

usage() {
  cat <<'EOF'
Usage:
  ./run_juqp_train_router.sh

Main defaults:
  CKPT=ckpt/model_w_juqp.pth
  CFG_FILE=cfgs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml
  SPLIT_FILE=../data/kitti_fov/ImageSets/train.txt
  QUANT_MAP='1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048'
  THRESHOLDS='0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075'

Step switches:
  RUN_TEST_SPLIT=1      # run test_split.py to create combo_*/result.pkl
  RUN_NEW_SPLIT=1       # run new_split.py to create AP matrix CSV
  RUN_JUQP_LABELS=1     # create one JUQP label CSV per threshold triple
  RUN_ROUTER_SPLIT=1    # split original train ids for proxy training/calibration
  RUN_ROUTER_TRAIN=1    # train train_cost_proxy.py and calibrate cost head

Examples:
  CUDA_VISIBLE_DEVICES=2 ./run_juqp_train_router.sh
  RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./run_juqp_train_router.sh
  ROUTER_VAL_RATIO=0.2 ROUTER_SHUFFLE=1 ./run_juqp_train_router.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_dir "$TOOLS_DIR" "OpenPCDet tools directory"
cd "$TOOLS_DIR"

require_file "$CFG_FILE" "OpenPCDet config"
require_file "$CKPT" "detection checkpoint"
require_file "$SPLIT_FILE" "original KITTI train split"

if [[ "$RUN_TEST_SPLIT" == "1" ]]; then
  require_dir "$MASK_DIR" "train foreground/background mask directory"
fi

log "JUQP train-label and router pipeline started"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "Config: ${CFG_FILE}"
log "Checkpoint: ${CKPT}"
log "Original train split: ${SPLIT_FILE}"
log "Quant map: ${QUANT_MAP}"
log "Thresholds: ${THRESHOLDS}"
log "AP CSV: ${AP_CSV}"
log "JUQP label dir: ${JUQP_LABEL_DIR}"
log "Router out dir: ${ROUTER_OUT_DIR}"

if [[ "$RUN_TEST_SPLIT" == "1" ]]; then
  log "Step 1/5: generate split-GPCC detection result.pkl files on original KITTI train"
  run_cmd env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" test_split.py \
    --cfg_file "$CFG_FILE" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$TEST_SPLIT_WORKERS" \
    --extra_tag "$EVAL_EXTRA_TAG" \
    --eval_tag "$EVAL_TAG" \
    --mask_dir "$MASK_DIR" \
    --quant_map "$QUANT_MAP"
else
  log "Step 1/5 skipped: RUN_TEST_SPLIT=0"
fi

if [[ "$RUN_NEW_SPLIT" == "1" ]]; then
  log "Step 2/5: calculate per-frame AP sensitivity matrix"
  require_dir "$EVAL_DIR" "test_split eval output directory"
  run_cmd "$PYTHON_BIN" new_split.py \
    --cfg_file "$CFG_FILE" \
    --split_file "$SPLIT_FILE" \
    --eval_dir "$EVAL_DIR" \
    --out_csv "$AP_CSV" \
    --workers "$NEW_SPLIT_WORKERS" \
    --quant_map "$QUANT_MAP"
  require_file "$AP_CSV" "AP matrix CSV"
else
  log "Step 2/5 skipped: RUN_NEW_SPLIT=0"
  require_file "$AP_CSV" "existing AP matrix CSV"
fi

if [[ "$RUN_JUQP_LABELS" == "1" ]]; then
  log "Step 3/5: derive JUQP labels for each threshold triple"
  mkdir -p "$JUQP_LABEL_DIR"
  IFS=';' read -r -a threshold_items <<< "$THRESHOLDS"
  for triple in "${threshold_items[@]}"; do
    triple="${triple// /}"
    [[ -n "$triple" ]] || continue
    IFS=',' read -r car_thr ped_thr cyc_thr <<< "$triple"
    [[ -n "${car_thr:-}" && -n "${ped_thr:-}" && -n "${cyc_thr:-}" ]] || die "Invalid threshold triple: ${triple}"
    out_csv="${JUQP_LABEL_DIR}/juqp_train_$(sanitize_threshold_name "$triple").csv"
    run_cmd "$PYTHON_BIN" jucp_split.py \
      --ap_csv "$AP_CSV" \
      --out_csv "$out_csv" \
      --car_threshold "$car_thr" \
      --ped_threshold "$ped_thr" \
      --cyc_threshold "$cyc_thr"
  done
else
  log "Step 3/5 skipped: RUN_JUQP_LABELS=0"
fi

if [[ "$RUN_ROUTER_SPLIT" == "1" ]]; then
  log "Step 4/5: split original KITTI train ids for proxy training and cost calibration"
  make_router_split
else
  log "Step 4/5 skipped: RUN_ROUTER_SPLIT=0"
  require_file "$ROUTER_TRAIN_SPLIT" "router train split"
  require_file "$ROUTER_VAL_SPLIT" "router validation split"
fi

if [[ "$RUN_ROUTER_TRAIN" == "1" ]]; then
  log "Step 5/5: train sparse cost proxy and calibrate cost head"
  require_file "$ROUTER_TRAIN_SPLIT" "router train split"
  require_file "$ROUTER_VAL_SPLIT" "router validation split"
  require_dir "$VELODYNE_DIR" "KITTI velodyne directory"

  run_cmd env OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" train_cost_proxy.py \
      --velodyne_dir "$VELODYNE_DIR" \
      --train_split "$ROUTER_TRAIN_SPLIT" \
      --ap_csv "$AP_CSV" \
      --val_split "$ROUTER_VAL_SPLIT" \
      --val_ap_csv "$AP_CSV" \
      --thresholds "$THRESHOLDS" \
      --test_every 0 \
      --out_dir "$ROUTER_OUT_DIR" \
      --epochs "$ROUTER_EPOCHS" \
      --batch_size "$ROUTER_BATCH_SIZE" \
      --workers "$ROUTER_WORKERS" \
      --voxel_size 0.16 0.16 0.16 \
      --point_cloud_range 0 -40 -3 70.4 40 1 \
      --max_voxels 50000 \
      --feat_dim 256 \
      --ap_drop_scale 100 \
      --lambda_threshold 0.1 \
      --ap_weights 10.0 1.0 1.0 \
      --lr 5e-4 \
      --jitter_std 0.005 \
      --weight_decay 5e-4 \
      --device "$ROUTER_DEVICE" \
      --calibrate_cost \
      --calibration_epochs "$CALIBRATION_EPOCHS" \
      --calibration_lr 1e-2
else
  log "Step 5/5 skipped: RUN_ROUTER_TRAIN=0"
fi

log "JUQP train-label and router pipeline finished"
log "AP CSV: ${TOOLS_DIR}/${AP_CSV}"
log "JUQP label dir: ${TOOLS_DIR}/${JUQP_LABEL_DIR}"
log "Router output: ${TOOLS_DIR}/${ROUTER_OUT_DIR}"
