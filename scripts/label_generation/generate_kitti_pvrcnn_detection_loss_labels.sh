#!/usr/bin/env bash
set -Eeuo pipefail

# Generate six absolute PV-RCNN loss targets for the KITTI camera-FOV point
# clouds.  This script intentionally does not train a routing network.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/OpenPCDet/tools"
EXPORT_SCRIPT="${ROOT_DIR}/scripts/label_generation/export_kitti_pvrcnn_detection_loss.py"
MERGE_SCRIPT="${ROOT_DIR}/scripts/label_generation/merge_kitti_pvrcnn_detection_loss_shards.py"
PYTHON_BIN="${PYTHON_BIN:-/home/sm/miniconda3/envs/SparsePCGC/bin/python}"

CFG_FILE="${CFG_FILE:-${ROOT_DIR}/integrations/openpcdet/configs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml}"
CKPT="${CKPT:-${TOOLS_DIR}/ckpt/model_non_reflectance.pth}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT_DIR}/OpenPCDet/data/kitti_fov/ImageSets/train.txt}"
FOV_VELODYNE_DIR="${FOV_VELODYNE_DIR:-${ROOT_DIR}/OpenPCDet/data/kitti_fov/training/velodyne}"

# The candidates are ordered coarse to fine.  The exported CSV retains both
# L0..L5 absolute total losses and deltas relative to the finest 64 mm level.
QUANT_MAP="${QUANT_MAP:-1/2048,1/2048;1/1024,1/1024;1/512,1/512;1/256,1/256;1/128,1/128;1/64,1/64}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-0,1,2,3,4,5}"
# Threshold-label CSVs are retained only for backward-compatible inspection;
# the current router is trained from the six absolute L*_total_loss columns.
LOSS_THRESHOLDS="${LOSS_THRESHOLDS:-0.0,0.05,0.10,0.20,0.40,0.80}"
LOSS_GPU_IDS="${LOSS_GPU_IDS:-0,1,2,3,4,5,6}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_results/kitti_pvrcnn_loss_labels}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
SHARD_ROOT="${SHARD_ROOT:-${RUN_DIR}/shards}"
LABEL_DIR="${LABEL_DIR:-${RUN_DIR}/labels}"
LOSS_CSV="${LOSS_CSV:-${RUN_DIR}/train_pvrcnn_six_scale_losses.csv}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUTPUT_ROOT}/archives/${RUN_ID}}"
PREFIX="${PREFIX:-kitti_pvrcnn_six_scale}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

wait_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || die "One or more PV-RCNN loss shards failed"
}

require_file "$PYTHON_BIN"
require_file "$CFG_FILE"
require_file "$CKPT"
require_file "$SPLIT_FILE"
require_dir "$FOV_VELODYNE_DIR"
require_file "$EXPORT_SCRIPT"
require_file "$MERGE_SCRIPT"

[[ ! -e "$RUN_DIR" ]] || die "Run directory already exists: $RUN_DIR"
[[ ! -e "$ARCHIVE_DIR" ]] || die "Archive directory already exists: $ARCHIVE_DIR"
mkdir -p "$SHARD_ROOT" "$LABEL_DIR"

TOTAL_FRAMES="$(grep -cve '^[[:space:]]*$' "$SPLIT_FILE")"
IFS=',' read -r -a LOSS_GPUS <<< "$LOSS_GPU_IDS"
NUM_SHARDS="${#LOSS_GPUS[@]}"
[[ "$TOTAL_FRAMES" -gt 0 ]] || die "Training split is empty: $SPLIT_FILE"
[[ "$NUM_SHARDS" -gt 0 ]] || die "LOSS_GPU_IDS is empty"

log "Generating six-scale PV-RCNN losses for ${TOTAL_FRAMES} KITTI FOV frames"
log "Detector checkpoint: ${CKPT}"
log "Quantization levels, coarse to fine: ${QUANT_MAP}"
log "GPUs: ${LOSS_GPU_IDS}"

BASE_SIZE=$((TOTAL_FRAMES / NUM_SHARDS))
REMAINDER=$((TOTAL_FRAMES % NUM_SHARDS))
START_INDEX=0
LOSS_PIDS=()

for ((SHARD_ID=0; SHARD_ID<NUM_SHARDS; SHARD_ID++)); do
  SHARD_SIZE="$BASE_SIZE"
  if ((SHARD_ID < REMAINDER)); then
    SHARD_SIZE=$((SHARD_SIZE + 1))
  fi
  GPU_ID="${LOSS_GPUS[$SHARD_ID]}"
  SHARD_DIR="${SHARD_ROOT}/shard_${SHARD_ID}"
  mkdir -p "${SHARD_DIR}/labels"
  log "shard=${SHARD_ID} gpu=${GPU_ID} start=${START_INDEX} frames=${SHARD_SIZE}"
  (
    cd "$TOOLS_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 "$PYTHON_BIN" \
      "$EXPORT_SCRIPT" \
      --cfg_file "$CFG_FILE" \
      --ckpt "$CKPT" \
      --split_file "$SPLIT_FILE" \
      --quant_map "$QUANT_MAP" \
      --candidate_labels "$CANDIDATE_LABELS" \
      --loss_thresholds "$LOSS_THRESHOLDS" \
      --out_dir "${SHARD_DIR}/labels" \
      --prefix "$PREFIX" \
      --loss_csv "${SHARD_DIR}/loss_sensitivity.csv" \
      --device cuda \
      --start_index "$START_INDEX" \
      --max_frames "$SHARD_SIZE"
  ) >"${SHARD_DIR}/run.log" 2>&1 &
  LOSS_PIDS+=("$!")
  START_INDEX=$((START_INDEX + SHARD_SIZE))
done

wait_jobs "${LOSS_PIDS[@]}"

cd "$TOOLS_DIR"
"$PYTHON_BIN" "$MERGE_SCRIPT" \
  --shard_root "$SHARD_ROOT" \
  --split_file "$SPLIT_FILE" \
  --out_dir "$LABEL_DIR" \
  --loss_csv "$LOSS_CSV" \
  --prefix "$PREFIX"

CSV_ROWS="$(awk 'END {print NR > 0 ? NR - 1 : 0}' "$LOSS_CSV")"
[[ "$CSV_ROWS" -eq "$TOTAL_FRAMES" ]] || \
  die "Merged CSV has ${CSV_ROWS} rows; expected ${TOTAL_FRAMES}"
HEADER="$(head -n 1 "$LOSS_CSV")"
for LEVEL in 0 1 2 3 4 5; do
  [[ ",$HEADER," == *",L${LEVEL}_total_loss,"* ]] || \
    die "Merged CSV is missing L${LEVEL}_total_loss"
done

mkdir -p "$ARCHIVE_DIR"
cp "$LOSS_CSV" "${ARCHIVE_DIR}/train_pvrcnn_six_scale_losses.csv"
cp "$SPLIT_FILE" "${ARCHIVE_DIR}/source_train_split.txt"
cp -a "$LABEL_DIR" "${ARCHIVE_DIR}/labels"
printf '%s\n' \
  "run_id=${RUN_ID}" \
  "frames=${TOTAL_FRAMES}" \
  "cfg=${CFG_FILE}" \
  "checkpoint=${CKPT}" \
  "quant_map=${QUANT_MAP}" \
  "candidate_labels=${CANDIDATE_LABELS}" \
  "loss_columns=L0_total_loss,...,L5_total_loss" \
  >"${ARCHIVE_DIR}/run_metadata.txt"
(
  cd "$ARCHIVE_DIR"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)

touch "${RUN_DIR}/LOSS_LABEL_GENERATION_COMPLETE"
log "Loss-label generation complete"
log "Primary CSV: ${LOSS_CSV}"
log "Manifest: ${LABEL_DIR}/${PREFIX}_manifest.json"
log "Verified archive: ${ARCHIVE_DIR}"
