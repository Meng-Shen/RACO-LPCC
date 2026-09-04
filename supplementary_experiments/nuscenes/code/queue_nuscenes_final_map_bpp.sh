#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR"

BASE="${RACO_NUSCENES_ROOT:-/home/sm/raco_rate_aware_nuscenes_20260822}"
ROOT="$BASE/experiments/nuscenes_rate_aware_from_kitti_rate_aware"
CACHE="$ROOT/val_prediction_cache_epoch6"
OUTPUT="$ROOT/final_map_bpp_epoch25"
PYTHON=/home/sm/miniconda3/envs/openmmlab/bin/python
MIM=/home/sm/miniconda3/envs/openmmlab/lib/python3.8/site-packages/mmdet3d/.mim

mkdir -p "$OUTPUT"
while [[ "$(find "$CACHE" -type f -name predictions.pkl | wc -l)" -lt 4 ]]; do
    sleep 30
done

export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
"$PYTHON" -u "$CODE/evaluate_nuscenes_rate_aware_map_bpp.py" \
    --config "$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_singleframe_recovery_12e_nus-3d.py" \
    --prediction-root "$CACHE" \
    --rate-aware-predictions-csv "$ROOT/final_eval_epoch25/test_rate_aware_predictions.csv" \
    --output-dir "$OUTPUT" \
    --checkpoint-epoch 25 \
    --parallel-workers 12
