#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/DATA/sm/RACO-LPCC
PYTHON=/home/sm/miniconda3/envs/SparsePCGC/bin/python
TOOLS=$ROOT/OpenPCDet/tools
ROUTER_EVAL_SCRIPT=$ROOT/scripts/curve_tools/evaluate_kitti_router_from_pkls.py
MERGE_CURVE_SCRIPT=$ROOT/scripts/curve_tools/merge_kitti_detection_router_curve.py
PLOT_SCRIPT=$ROOT/scripts/curve_tools/plot_kitti_gpcc_residual_map_bpp_curves.py
OUT=$ROOT/experiment_results/gpcc_current_q_ones_scratch_q128dist100_lrproxy_router_20260901
OLD=$ROOT/experiment_results/kitti_detection_lrproxy_pvrcnn_zero_shot_20260829
MANIFEST=$OLD/labels/lrproxy_manifest.json
OLD_BPP=$OLD/pvrcnn/gpcc/router_average_results.csv
FIXED_LINKS=$OUT/fixed_result_links
AP=$OUT/pvrcnn/original_gpcc_route_then_residual_ap.csv
CURVE=$OUT/pvrcnn/original_gpcc_route_then_residual_curve.csv
QUANT_MAP='1/2048,1/2048;1/1024,1/1024;1/512,1/512;1/256,1/256;1/128,1/128;1/64,1/64'

cd "$TOOLS"
"$PYTHON" "$ROUTER_EVAL_SCRIPT" \
  --cfg_file "$ROOT/integrations/openpcdet/configs/kitti_models/pv_rcnn_fov_geometry.yaml" \
  --eval_dir "$FIXED_LINKS" --quant_map "$QUANT_MAP" \
  --manifest "$MANIFEST" --out "$AP" \
  --save_mixed_pkls_dir "$OUT/pvrcnn/original_route_then_residual_mixed_pkls"

cd "$ROOT"
"$PYTHON" "$MERGE_CURVE_SCRIPT" --ap_csv "$AP" --gpcc_csv "$OLD_BPP" --out "$CURVE"
"$PYTHON" "$PLOT_SCRIPT" \
  --fixed-csv "$ROOT/experiment_results/gpcc_current_q_ones_scratch_q128dist100_20260901/comparison/scratch_q128dist100_map_bpp.csv" \
  --plain-router-csv "$OLD/pvrcnn/router_gpcc_curve.csv" \
  --residual-router-csv "$OUT/pvrcnn/router_gpcc_residual_curve.csv" \
  --plain-route-then-residual-csv "$CURVE" \
  --output-dir "$OUT/comparison"
