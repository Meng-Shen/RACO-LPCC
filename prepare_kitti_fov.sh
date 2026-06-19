#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/OpenPCDet/data/kitti_fov}"
PYTHON_BIN="${PYTHON_BIN:-/home/sm/miniconda3/envs/SparsePCGC/bin/python}"
WORKERS="${WORKERS:-8}"

"${PYTHON_BIN}" "${ROOT_DIR}/OpenPCDet/tools/prepare_kitti_fov_dataset.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}"

echo
echo "FOV-only KITTI dataset is ready: ${OUTPUT_ROOT}"
echo "Detection config: cfgs/kitti_models/pv_rcnn_fov_geometry.yaml"
echo "Train-as-test config: cfgs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml"
