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
DET_CKPT="${DET_CKPT:-ckpt/model_non_reflectance.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-16}"
DET_EXTRA_TAG="${DET_EXTRA_TAG:-default}"
DET_EVAL_TAG="${DET_EVAL_TAG:-default}"

KITTI_ROOT="${KITTI_ROOT:-${SCRIPT_DIR}/OpenPCDet/data/kitti_fov}"
KITTI_VELODYNE="${KITTI_VELODYNE:-${KITTI_ROOT}/training/velodyne}"
SPLIT_FILE="${SPLIT_FILE:-${KITTI_ROOT}/ImageSets/val.txt}"
GPCC_CFG="${GPCC_CFG:-${SCRIPT_DIR}/extention/kitti.cfg}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/point_pairs/verify_juqp_fov}"
GPCC_RESULTS_DIR="${GPCC_RESULTS_DIR:-${SCRIPT_DIR}/point_pairs/baseline_fov/gpcc}"
GPCC_TMP_DIR="${GPCC_TMP_DIR:-${OUT_DIR}/tmp_gpcc}"
BASELINE_AP_CSV="${BASELINE_AP_CSV:-${SCRIPT_DIR}/point_pairs/baseline_fov/baseline_ap.csv}"
BASELINE_CURVE_CSV="${BASELINE_CURVE_CSV:-${OUT_DIR}/baseline_gpcc_curve.csv}"
AP_SENSITIVITY_CSV="${AP_SENSITIVITY_CSV:-${OUT_DIR}/val_ap_sensitivity.csv}"
ORACLE_OUT_DIR="${ORACLE_OUT_DIR:-${OUT_DIR}/oracle_juqp}"
PLOT_DIR="${PLOT_DIR:-${OUT_DIR}/plots}"
PKL_EVAL_DIR="${PKL_EVAL_DIR:-${OUT_DIR}/gpcc_scale_eval_links}"

EXISTING_GPCC_CSV="${EXISTING_GPCC_CSV:-${SCRIPT_DIR}/point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv}"
DEFAULT_SCALES="1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512"
SCALES="${SCALES:-}"
LAGRANGE_LAMBDAS="${LAGRANGE_LAMBDAS:-0,0.00025,0.0005,0.001,0.002,0.004,0.008,0.016,0.032}"
OBJECTIVE="${OBJECTIVE:-Car}"

RUN_AP="${RUN_AP:-0}"
RUN_GPCC="${RUN_GPCC:-0}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

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

read_scales_from_gpcc_csv() {
  local csv_path="$1"
  "$PYTHON_BIN" - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
scales = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        value = row.get("posQuantscale") or row.get("scale")
        if value not in (None, ""):
            scales.append(str(value))

if not scales:
    raise SystemExit(f"No posQuantscale/scale values found in {path}")
print(",".join(scales))
PY
}

make_quant_map() {
  local scales="$1"
  "$PYTHON_BIN" - "$scales" <<'PY'
import sys

def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        n, d = value.split("/", 1)
        return float(n) / float(d)
    return float(value)

items = []
for item in sys.argv[1].split(","):
    item = item.strip()
    if not item:
        continue
    scale = parse_scale(item)
    items.append(f"{scale:.12g},{scale:.12g}")

if len(items) < 2:
    raise SystemExit("At least two scales are required for JUQP verification")
print(";".join(items))
PY
}

link_scale_results_as_combos() {
  local source_eval_dir="$1"
  local link_eval_dir="$2"
  local scales="$3"
  "$PYTHON_BIN" - "$source_eval_dir" "$link_eval_dir" "$scales" <<'PY'
import os
import sys
from pathlib import Path

def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        n, d = value.split("/", 1)
        return float(n) / float(d)
    return float(value)

source_eval_dir = Path(sys.argv[1])
link_eval_dir = Path(sys.argv[2])
scales = [parse_scale(item) for item in sys.argv[3].split(",") if item.strip()]
link_eval_dir.mkdir(parents=True, exist_ok=True)
for idx, scale in enumerate(scales):
    src = source_eval_dir / f"scale_{scale}" / "result.pkl"
    dst_dir = link_eval_dir / f"combo_{idx}_fg_{scale:.6f}_bg_{scale:.6f}"
    dst = dst_dir / "result.pkl"
    if not src.exists():
        raise SystemExit(f"Missing result.pkl from test_pos.py: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(os.path.relpath(src, dst_dir))
    except OSError:
        import shutil
        shutil.copy2(src, dst)
    print(f"{dst} -> {src}")
PY
}

OUT_DIR="$(abs_path "$OUT_DIR")"
GPCC_RESULTS_DIR="$(abs_path "$GPCC_RESULTS_DIR")"
GPCC_TMP_DIR="$(abs_path "$GPCC_TMP_DIR")"
BASELINE_AP_CSV="$(abs_path "$BASELINE_AP_CSV")"
BASELINE_CURVE_CSV="$(abs_path "$BASELINE_CURVE_CSV")"
AP_SENSITIVITY_CSV="$(abs_path "$AP_SENSITIVITY_CSV")"
ORACLE_OUT_DIR="$(abs_path "$ORACLE_OUT_DIR")"
PLOT_DIR="$(abs_path "$PLOT_DIR")"
PKL_EVAL_DIR="$(abs_path "$PKL_EVAL_DIR")"
KITTI_ROOT="$(abs_path "$KITTI_ROOT")"
KITTI_VELODYNE="$(abs_path "$KITTI_VELODYNE")"
SPLIT_FILE="$(abs_path "$SPLIT_FILE")"
GPCC_CFG="$(abs_path "$GPCC_CFG")"
EXISTING_GPCC_CSV="$(abs_path "$EXISTING_GPCC_CSV")"

mkdir -p "$OUT_DIR" "$GPCC_RESULTS_DIR" "$GPCC_TMP_DIR" "$ORACLE_OUT_DIR" "$PLOT_DIR" "$PKL_EVAL_DIR"
require_dir "$OPENPCDET_TOOLS"

if [[ -z "$SCALES" ]]; then
  if [[ -f "$EXISTING_GPCC_CSV" ]]; then
    SCALES="$(read_scales_from_gpcc_csv "$EXISTING_GPCC_CSV")"
  else
    SCALES="$DEFAULT_SCALES"
  fi
fi
QUANT_MAP="${QUANT_MAP:-$(make_quant_map "$SCALES")}"

CFG_STEM="$(basename "${CFG_FILE%.yaml}")"
DET_EVAL_DIR="${DET_EVAL_DIR:-${SCRIPT_DIR}/OpenPCDet/output/kitti_models/${CFG_STEM}/${DET_EXTRA_TAG}/eval/epoch_no_number/val/${DET_EVAL_TAG}}"
DET_EVAL_DIR="$(abs_path "$DET_EVAL_DIR")"
case "$CFG_FILE" in
  /*) ORACLE_CFG_FILE="$CFG_FILE" ;;
  *) ORACLE_CFG_FILE="${OPENPCDET_TOOLS}/${CFG_FILE}" ;;
esac

log "Using scales: $SCALES"
log "Using quant_map for oracle/JUQP: $QUANT_MAP"

if [[ ! -d "$KITTI_VELODYNE" || ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  log "FOV-only KITTI data is missing; generating it first"
  OUTPUT_ROOT="$KITTI_ROOT" "$SCRIPT_DIR/prepare_kitti_fov.sh"
fi

require_dir "$KITTI_VELODYNE"
require_file "$SPLIT_FILE"
require_file "$GPCC_CFG"

if [[ "$RUN_AP" == "1" ]]; then
  log "Step 1/5: run GPCC baseline detection by direct global quantization"
  cd "$OPENPCDET_TOOLS"
  require_file "$CFG_FILE"
  require_file "$DET_CKPT"
  "$PYTHON_BIN" test_pos.py \
    --cfg_file "$CFG_FILE" \
    --ckpt "$DET_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --extra_tag "$DET_EXTRA_TAG" \
    --eval_tag "$DET_EVAL_TAG" \
    --scales "$SCALES"

  AP_LOG="$(find "$DET_EVAL_DIR" -name 'log_eval_pos_*.txt' 2>/dev/null | sort | tail -n 1)"
  require_file "$AP_LOG"
  "$PYTHON_BIN" parse_baseline_ap_log.py --log "$AP_LOG" --out "$BASELINE_AP_CSV"
else
  log "Step 1/5 skipped: RUN_AP=0"
  require_file "$BASELINE_AP_CSV"
  require_dir "$DET_EVAL_DIR"
fi

if [[ "$RUN_SENSITIVITY" == "1" || "$RUN_ORACLE" == "1" ]]; then
  log "Preparing combo-style result.pkl links for per-frame AP/oracle scripts"
  link_scale_results_as_combos "$DET_EVAL_DIR" "$PKL_EVAL_DIR" "$SCALES"
fi

if [[ "$RUN_GPCC" == "1" ]]; then
  log "Step 2/5: measure GPCC bpp/time for the same scales"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" GPCC/baseline_rates.py \
    --testdata "$KITTI_VELODYNE" \
    --split_file "$SPLIT_FILE" \
    --scales "$SCALES" \
    --results "$GPCC_RESULTS_DIR" \
    --tmp_dir "$GPCC_TMP_DIR" \
    --cfg "$GPCC_CFG"
else
  log "Step 2/5 skipped: RUN_GPCC=0"
  require_file "${GPCC_RESULTS_DIR}/gpcc_baseline_average.csv"
  require_file "${GPCC_RESULTS_DIR}/gpcc_baseline_details.csv"
fi

log "Merging fixed-rate GPCC AP-bpp curve"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" merge_baseline_curve.py \
  --ap_csv "$BASELINE_AP_CSV" \
  --gpcc_csv "${GPCC_RESULTS_DIR}/gpcc_baseline_average.csv" \
  --out "$BASELINE_CURVE_CSV"

if [[ "$RUN_SENSITIVITY" == "1" ]]; then
  log "Step 3/5: compute per-frame AP sensitivity from result.pkl files"
  cd "$OPENPCDET_TOOLS"
  "$PYTHON_BIN" new_split.py \
    --cfg_file "$CFG_FILE" \
    --split_file "$SPLIT_FILE" \
    --eval_dir "$PKL_EVAL_DIR" \
    --out_csv "$AP_SENSITIVITY_CSV" \
    --workers "$WORKERS" \
    --quant_map "$QUANT_MAP"
else
  log "Step 3/5 skipped: RUN_SENSITIVITY=0"
  if [[ "$RUN_ORACLE" == "1" ]]; then
    require_file "$AP_SENSITIVITY_CSV"
  fi
fi

if [[ "$RUN_ORACLE" == "1" ]]; then
  log "Step 4/5: derive oracle JUQP labels and ideal AP-bpp curve"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" compute_oracle_router_curve.py \
    --cfg_file "$ORACLE_CFG_FILE" \
    --eval_dir "$PKL_EVAL_DIR" \
    --ap_csv "$AP_SENSITIVITY_CSV" \
    --split_details_csv "${GPCC_RESULTS_DIR}/gpcc_baseline_details.csv" \
    --quant_map "$QUANT_MAP" \
    --objective "$OBJECTIVE" \
    --lambdas "$LAGRANGE_LAMBDAS" \
    --out_dir "$ORACLE_OUT_DIR" \
    --save_mixed_pkls_dir "${ORACLE_OUT_DIR}/mixed_result_pkls"
else
  log "Step 4/5 skipped: RUN_ORACLE=0"
  if [[ "$RUN_PLOT" == "1" ]]; then
    require_file "${ORACLE_OUT_DIR}/oracle_router_curve.csv"
  fi
fi

if [[ "$RUN_PLOT" == "1" ]]; then
  log "Step 5/5: plot GPCC baseline and oracle JUQP AP-bpp curves"
  cd "$SCRIPT_DIR"
  "$PYTHON_BIN" - "$BASELINE_CURVE_CSV" "${ORACLE_OUT_DIR}/oracle_router_curve.csv" "$PLOT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

baseline_csv = Path(sys.argv[1])
oracle_csv = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

classes = {
    "Car": "Car_3d_AP_R40_moderate",
    "Pedestrian": "Pedestrian_3d_AP_R40_moderate",
    "Cyclist": "Cyclist_3d_AP_R40_moderate",
}

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def points(rows, y_col):
    vals = []
    for row in rows:
        if row.get("bpp") in (None, "") or row.get(y_col) in (None, ""):
            continue
        vals.append((float(row["bpp"]), float(row[y_col])))
    return sorted(vals)

baseline_rows = read_rows(baseline_csv)
oracle_rows = read_rows(oracle_csv)

for cls, y_col in classes.items():
    b = points(baseline_rows, y_col)
    o = points(oracle_rows, y_col)
    if not b or not o:
        raise SystemExit(f"No plottable points for {cls}")

    plt.figure(figsize=(8.5, 6))
    plt.plot([x for x, _ in b], [y for _, y in b], color="#ff7f0e", marker="X",
             linestyle="--", linewidth=2.2, markersize=6, label="GPCC Baseline")
    plt.plot([x for x, _ in o], [y for _, y in o], color="#2ca02c", marker="o",
             linestyle="-", linewidth=2.2, markersize=6, label="Oracle JUQP")
    ys = [y for _, y in b + o]
    pad = max((max(ys) - min(ys)) * 0.08, 1.0)
    plt.ylim(max(0, min(ys) - pad), max(ys) + pad)
    plt.xlabel("Bits Per Point (bpp)", fontsize=13)
    plt.ylabel(f"{cls} 3D AP R40 Moderate (%)", fontsize=13)
    plt.title(f"{cls} AP-bpp Curve", fontsize=15, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"ap_bpp_{cls.lower()}.png"
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(out_path)
PY
else
  log "Step 5/5 skipped: RUN_PLOT=0"
fi

log "Done"
log "Baseline curve CSV: $BASELINE_CURVE_CSV"
log "Per-frame AP sensitivity CSV: $AP_SENSITIVITY_CSV"
log "Oracle JUQP curve CSV: ${ORACLE_OUT_DIR}/oracle_router_curve.csv"
log "Oracle JUQP labels: ${ORACLE_OUT_DIR}/oracle_rate_*_labels.csv"
log "Plots: $PLOT_DIR"
