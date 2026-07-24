#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
UNICORN_PYTHON_BIN="${UNICORN_PYTHON_BIN:-$PYTHON_BIN}"
UNICORN_ROOT="${UNICORN_ROOT:-/public/DATA/sm/Unicorn}"

KITTI_ROOT="${KITTI_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SOURCE_SPLIT_FILE="${SOURCE_SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"
TRAIN_POSQ="${TRAIN_POSQ:-64.0}"
RATES="${RATES:-0:0:1,0:1:2,0:1:4,0:1:8,0:1:16,0:1:32}"
RESOLUTION="${RESOLUTION:-80000}"
UNICORN_RESUME="${UNICORN_RESUME:-0}"

UNICORN_MODEL_DIR="${UNICORN_MODEL_DIR:-${ROOT_DIR}/unicorn/checkpoints}"
SR_CKPT="${SR_CKPT:-${UNICORN_MODEL_DIR}/sr/epoch_last.pth}"
LOSSLESS_LOW_CKPT="${LOSSLESS_LOW_CKPT:-${UNICORN_MODEL_DIR}/lossless/epoch_last.pth}"

CHANNELS="${CHANNELS:-32}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
BLOCK_LAYERS="${BLOCK_LAYERS:-3}"
BLOCK_TYPE="${BLOCK_TYPE:-conv}"

BASELINE_DETAIL_CSV="${BASELINE_DETAIL_CSV:-${ROOT_DIR}/point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv}"
SPLIT_DETAIL_CSV="${SPLIT_DETAIL_CSV:-${ROOT_DIR}/point_pairs/split_gpcc_fov/gpcc/split_all_details.csv}"
RENO_DETAIL_CSV="${RENO_DETAIL_CSV:-${ROOT_DIR}/point_pairs/reno_fov/reno/reno_details.csv}"
BASELINE_PSNR_CSV="${BASELINE_PSNR_CSV:-${ROOT_DIR}/point_pairs/psnr_bpp/baseline_psnr_details.csv}"
SPLIT_PSNR_CSV="${SPLIT_PSNR_CSV:-${ROOT_DIR}/point_pairs/psnr_bpp/split_psnr_details.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/point_pairs/unicorn_first10}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Runs Unicorn on the first 10 non-empty frame ids in SOURCE_SPLIT_FILE
(KITTI FOV val.txt by default), then aggregates first-10-frame PSNR-bpp curves.

Environment overrides:
  SOURCE_SPLIT_FILE, RATES, OUT_DIR, PYTHON_BIN, UNICORN_PYTHON_BIN,
  UNICORN_ROOT, KITTI_VELODYNE, SR_CKPT, LOSSLESS_LOW_CKPT,
  BASELINE_DETAIL_CSV, SPLIT_DETAIL_CSV, RENO_DETAIL_CSV,
  BASELINE_PSNR_CSV, SPLIT_PSNR_CSV, TRAIN_POSQ, RESOLUTION, UNICORN_RESUME
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

KITTI_VELODYNE="$(abs_path "$KITTI_VELODYNE")"
SOURCE_SPLIT_FILE="$(abs_path "$SOURCE_SPLIT_FILE")"
UNICORN_ROOT="$(abs_path "$UNICORN_ROOT")"
SR_CKPT="$(abs_path "$SR_CKPT")"
LOSSLESS_LOW_CKPT="$(abs_path "$LOSSLESS_LOW_CKPT")"
BASELINE_DETAIL_CSV="$(abs_path "$BASELINE_DETAIL_CSV")"
SPLIT_DETAIL_CSV="$(abs_path "$SPLIT_DETAIL_CSV")"
RENO_DETAIL_CSV="$(abs_path "$RENO_DETAIL_CSV")"
BASELINE_PSNR_CSV="$(abs_path "$BASELINE_PSNR_CSV")"
SPLIT_PSNR_CSV="$(abs_path "$SPLIT_PSNR_CSV")"
OUT_DIR="$(abs_path "$OUT_DIR")"

require_dir "$KITTI_VELODYNE"
require_dir "$UNICORN_ROOT"
require_file "$SOURCE_SPLIT_FILE"
require_file "$SR_CKPT"
require_file "$LOSSLESS_LOW_CKPT"
require_file "$BASELINE_DETAIL_CSV"
require_file "$SPLIT_DETAIL_CSV"
require_file "$RENO_DETAIL_CSV"
require_file "$BASELINE_PSNR_CSV"
require_file "$SPLIT_PSNR_CSV"

mapfile -t FRAME_IDS < <(awk 'NF {print $1; count += 1; if (count == 10) exit}' "$SOURCE_SPLIT_FILE")
if [[ ${#FRAME_IDS[@]} -ne 10 ]]; then
  echo "Expected at least 10 non-empty frame ids in $SOURCE_SPLIT_FILE, found ${#FRAME_IDS[@]}" >&2
  exit 1
fi

for frame_id in "${FRAME_IDS[@]}"; do
  require_file "${KITTI_VELODYNE}/${frame_id}.bin"
done

UNICORN_RESULTS_DIR="${OUT_DIR}/unicorn"
UNICORN_TMP_DIR="${OUT_DIR}/tmp"
UNICORN_BITSTREAM_DIR="${OUT_DIR}/bitstreams"
UNICORN_DECODED_DIR="${OUT_DIR}/decoded"
FIRST10_SPLIT_FILE="${OUT_DIR}/first10_split.txt"
mkdir -p "$OUT_DIR" "$UNICORN_RESULTS_DIR" "$UNICORN_TMP_DIR" \
  "$UNICORN_BITSTREAM_DIR" "$UNICORN_DECODED_DIR"
printf '%s\n' "${FRAME_IDS[@]}" > "$FIRST10_SPLIT_FILE"

cd "$ROOT_DIR"

echo "[i] Source split: $SOURCE_SPLIT_FILE"
echo "[i] First 10 frames: ${FRAME_IDS[*]}"
echo "[i] Rates: $RATES"
echo "[i] Output: $OUT_DIR"

UNICORN_EXTRA_ARGS=()
if [[ "$UNICORN_RESUME" == "1" ]]; then
  UNICORN_EXTRA_ARGS+=(--resume)
fi

"$UNICORN_PYTHON_BIN" unicorn/unicorn_rates_direct.py \
  --unicorn_root "$UNICORN_ROOT" \
  --testdata "$KITTI_VELODYNE" \
  --split_file "$FIRST10_SPLIT_FILE" \
  --train_posq "$TRAIN_POSQ" \
  --results "$UNICORN_RESULTS_DIR" \
  --tmp_dir "$UNICORN_TMP_DIR" \
  --bitstream_dir "$UNICORN_BITSTREAM_DIR" \
  --decoded_dir "$UNICORN_DECODED_DIR" \
  --rates "$RATES" \
  --ckptdir_low "$LOSSLESS_LOW_CKPT" \
  --ckptdir_sr_low "$SR_CKPT" \
  --disable_offset \
  --channels "$CHANNELS" \
  --kernel_size "$KERNEL_SIZE" \
  --block_layers "$BLOCK_LAYERS" \
  --block_type "$BLOCK_TYPE" \
  --resolution "$RESOLUTION" \
  "${UNICORN_EXTRA_ARGS[@]}"

"$PYTHON_BIN" unicorn/plot_first10_gpcc_reno_unicorn.py \
  --split_file "$FIRST10_SPLIT_FILE" \
  --gpcc_detail_csv "$BASELINE_DETAIL_CSV" \
  --gpcc_psnr_csv "$BASELINE_PSNR_CSV" \
  --split_detail_csv "$SPLIT_DETAIL_CSV" \
  --split_psnr_csv "$SPLIT_PSNR_CSV" \
  --reno_detail_csv "$RENO_DETAIL_CSV" \
  --unicorn_detail_csv "$UNICORN_RESULTS_DIR/unicorn_details.csv" \
  --out_dir "$OUT_DIR"

echo "[+] First-10-frame Unicorn test complete"
echo "[+] Average: ${UNICORN_RESULTS_DIR}/unicorn_average.csv"
echo "[+] Combined CSV: ${OUT_DIR}/first10_gpcc_reno_unicorn_psnr_bpp.csv"
