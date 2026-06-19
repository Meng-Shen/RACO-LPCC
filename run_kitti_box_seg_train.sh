#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_DIR="${ROOT_DIR}/mmdetection3d"
export PYTHONPATH="${MMDET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
KITTI_ROOT="${KITTI_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti_fov}"
DATA_ROOT="${DATA_ROOT:-${MMDET_DIR}/data/kitti_fov_box_seg}"
DEFAULT_PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
PRETRAINED="${PRETRAINED:-${MMDET_DIR}/ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth}"
VAL_RATIO="${VAL_RATIO:-0.10}"
SEED="${SEED:-2026}"
WORK_DIR="${WORK_DIR:-${MMDET_DIR}/work_dirs/minkunet_kitti_fov_box_seg_geometry}"
FG_WEIGHT="${FG_WEIGHT:-8.0}"
FORCE_REGENERATE_LABELS="${FORCE_REGENERATE_LABELS:-0}"

if [[ ! -d "${KITTI_ROOT}/training/velodyne" ||
      ! -f "${KITTI_ROOT}/fov_crop_stats.csv" ]]; then
  echo "FOV-only KITTI data is missing; generating it first."
  OUTPUT_ROOT="${KITTI_ROOT}" "${ROOT_DIR}/prepare_kitti_fov.sh"
fi

for path in \
  "${KITTI_ROOT}/ImageSets/train.txt" \
  "${KITTI_ROOT}/training/velodyne" \
  "${KITTI_ROOT}/training/label_2" \
  "${KITTI_ROOT}/training/calib" \
  "${KITTI_ROOT}/training/image_2" \
  "${PRETRAINED}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${DATA_ROOT}"

LABEL_CACHE="${DATA_ROOT}/label_cache.meta"
EXPECTED_CACHE="$(
  printf 'kitti_root=%s\nval_ratio=%s\nseed=%s\nclasses=Car,Pedestrian,Cyclist\n' \
    "$(realpath "${KITTI_ROOT}")" "${VAL_RATIO}" "${SEED}"
)"
SOURCE_COUNT="$(awk 'NF {count++} END {print count+0}' \
  "${KITTI_ROOT}/ImageSets/train.txt")"
LABEL_COUNT=0
if [[ -d "${DATA_ROOT}/box_seg_labels" ]]; then
  LABEL_COUNT="$(
    find "${DATA_ROOT}/box_seg_labels" -maxdepth 1 -type f -name '*.label' |
      wc -l
  )"
fi

LABEL_CACHE_VALID=0
if [[ "${FORCE_REGENERATE_LABELS}" != "1" &&
      -f "${LABEL_CACHE}" &&
      -f "${DATA_ROOT}/kitti_box_seg_infos_train.pkl" &&
      -f "${DATA_ROOT}/kitti_box_seg_infos_val.pkl" &&
      "${LABEL_COUNT}" -ge "${SOURCE_COUNT}" &&
      "$(cat "${LABEL_CACHE}")" == "${EXPECTED_CACHE}" ]]; then
  LABEL_CACHE_VALID=1
fi

if [[ "${LABEL_CACHE_VALID}" == "1" ]]; then
  echo "[1/3] Reusing ${LABEL_COUNT} cached KITTI point-label files"
  echo "      Cache directory: ${DATA_ROOT}"
else
  echo "[1/3] Generating and caching KITTI box-supervised point labels"
  GENERATE_ARGS=(
    --kitti-root "${KITTI_ROOT}"
    --output-root "${DATA_ROOT}"
    --source-split train
    --val-ratio "${VAL_RATIO}"
    --seed "${SEED}"
  )
  if [[ "${FORCE_REGENERATE_LABELS}" == "1" ]]; then
    GENERATE_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" "${MMDET_DIR}/tools/create_kitti_box_seg.py" \
    "${GENERATE_ARGS[@]}"
  printf '%s\n' "${EXPECTED_CACHE}" > "${LABEL_CACHE}"
fi

if [[ ! -e "${DATA_ROOT}/training" ]]; then
  ln -s "${KITTI_ROOT}/training" "${DATA_ROOT}/training"
fi

echo "[2/3] Training geometry-only MinkUNet and validating every epoch"
cd "${MMDET_DIR}"
"${PYTHON_BIN}" tools/train_geometry_only.py \
  configs/minkunet/minkunet34_w32_minkowski_geometry_kitti_box_seg.py \
  --pretrained "${PRETRAINED}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    "train_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "val_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "test_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "model.decode_head.loss_decode.class_weight=[1.0,${FG_WEIGHT}]"

BEST_LINK="${WORK_DIR}/best_foreground.pth"
if [[ ! -e "${BEST_LINK}" ]]; then
  BEST_LINK="$(find "${WORK_DIR}" -maxdepth 1 -type f \
    -name 'best_foreground*.pth' -printf '%T@ %p\n' |
    sort -nr | head -n1 | cut -d' ' -f2- || true)"
fi
if [[ -z "${BEST_LINK}" || ! -e "${BEST_LINK}" ]]; then
  echo "Training finished, but no best-foreground checkpoint was found." >&2
  exit 1
fi

echo "[3/3] Done"
echo "Best validation-foreground-IoU checkpoint: ${BEST_LINK}"
