#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti_fov}"
PYTHON_BIN="${PYTHON_BIN:-/home/sm/miniconda3/envs/SparsePCGC/bin/python}"
WORKERS="${WORKERS:-8}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data_preparation/prepare_kitti_fov_dataset.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}"

echo
echo "FOV-only KITTI dataset is ready: ${OUTPUT_ROOT}"
echo "Detection config: integrations/openpcdet/configs/kitti_models/pv_rcnn_fov_geometry.yaml"
