#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
RENO_PYTHON_BIN="${RENO_PYTHON_BIN:-$PYTHON_BIN}"
RENO_ROOT="${RENO_ROOT:-/public/DATA/sm/RENO}"
KITTI_ROOT="${KITTI_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SCALES="${SCALES:-1/64,1/128,1/256,1/512,1/1024,1/2048}"
RESOLUTION="${RESOLUTION:-80000}"
DEVICE="${DEVICE:-cuda}"

BASELINE_DETAIL_CSV="${BASELINE_DETAIL_CSV:-${ROOT_DIR}/point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv}"
SPLIT_DETAIL_CSV="${SPLIT_DETAIL_CSV:-${ROOT_DIR}/point_pairs/split_gpcc_fov/gpcc/split_all_details.csv}"
BASELINE_PSNR_CSV="${BASELINE_PSNR_CSV:-${ROOT_DIR}/point_pairs/psnr_bpp/baseline_psnr_details.csv}"
SPLIT_PSNR_CSV="${SPLIT_PSNR_CSV:-${ROOT_DIR}/point_pairs/psnr_bpp/split_psnr_details.csv}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/point_pairs/reno_single_frame}"

usage() {
  cat <<EOF
Usage: $(basename "$0") RENO_CKPT FRAME_ID_OR_BIN

Examples:
  reno/run_single_frame_psnr_bpp.sh point_pairs/reno_fov/model/ckpt.pt 000001
  reno/run_single_frame_psnr_bpp.sh /path/to/ckpt.pt OpenPCDet/data/kitti_fov/training/velodyne/000001.bin

Environment overrides:
  SCALES, OUT_ROOT, PYTHON_BIN, RENO_PYTHON_BIN, RENO_ROOT, KITTI_VELODYNE,
  BASELINE_DETAIL_CSV, SPLIT_DETAIL_CSV, BASELINE_PSNR_CSV, SPLIT_PSNR_CSV,
  RESOLUTION, DEVICE
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

RENO_CKPT="$1"
FRAME_INPUT="$2"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

RENO_CKPT="$(abs_path "$RENO_CKPT")"
require_file "$RENO_CKPT"
require_file "$BASELINE_DETAIL_CSV"
require_file "$SPLIT_DETAIL_CSV"
require_file "$BASELINE_PSNR_CSV"
require_file "$SPLIT_PSNR_CSV"

if [[ -f "$FRAME_INPUT" ]]; then
  FRAME_BIN="$(abs_path "$FRAME_INPUT")"
  FRAME_ID="$(basename "$FRAME_BIN" .bin)"
else
  FRAME_ID="$(basename "$FRAME_INPUT" .bin)"
  if [[ "$FRAME_ID" =~ ^[0-9]+$ ]]; then
    FRAME_ID="$(printf '%06d' "$FRAME_ID")"
  fi
  FRAME_BIN="${KITTI_VELODYNE}/${FRAME_ID}.bin"
fi
require_file "$FRAME_BIN"

OUT_DIR="${OUT_ROOT}/${FRAME_ID}"
RENO_RESULTS_DIR="${OUT_DIR}/reno"
RENO_TMP_DIR="${OUT_DIR}/tmp"
RENO_BITSTREAM_DIR="${OUT_DIR}/bitstreams"
mkdir -p "$OUT_DIR" "$RENO_RESULTS_DIR" "$RENO_TMP_DIR" "$RENO_BITSTREAM_DIR"

cd "$ROOT_DIR"

echo "[i] Frame: $FRAME_ID"
echo "[i] Point cloud: $FRAME_BIN"
echo "[i] RENO checkpoint: $RENO_CKPT"
echo "[i] Output: $OUT_DIR"

"$RENO_PYTHON_BIN" reno/reno_rates.py \
  --reno_root "$RENO_ROOT" \
  --testdata "$FRAME_BIN" \
  --scales "$SCALES" \
  --ckpt "$RENO_CKPT" \
  --results "$RENO_RESULTS_DIR" \
  --tmp_dir "$RENO_TMP_DIR" \
  --bitstream_dir "$RENO_BITSTREAM_DIR" \
  --kitti_root "$KITTI_ROOT" \
  --resolution "$RESOLUTION" \
  --device "$DEVICE"

"$PYTHON_BIN" reno/single_frame_psnr_bpp.py \
  --frame "$FRAME_ID" \
  --reno_detail_csv "$RENO_RESULTS_DIR/reno_details.csv" \
  --baseline_detail_csv "$BASELINE_DETAIL_CSV" \
  --split_detail_csv "$SPLIT_DETAIL_CSV" \
  --baseline_psnr_csv "$BASELINE_PSNR_CSV" \
  --split_psnr_csv "$SPLIT_PSNR_CSV" \
  --out_dir "$OUT_DIR"

echo "[+] Done"
