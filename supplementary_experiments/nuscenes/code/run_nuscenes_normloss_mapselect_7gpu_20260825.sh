#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${RACO_NUSCENES_ROOT:-/home/sm/raco_rate_aware_nuscenes_20260822}"
ENV=/home/sm/miniconda3/envs/openmmlab
PYTHON="$ENV/bin/python"
MIM="$ENV/lib/python3.8/site-packages/mmdet3d/.mim"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR"
CONFIG_ROOT="$SCRIPT_DIR/../configs"
DATA="$BASE/data/nuscenes"
SOURCE_TAG=multisweep_cbgs_bestmap_epoch4_router_7gpu_20260825
TAG=multisweep_cbgs_bestmap_epoch4_normloss_mapselect_7gpu_20260825
CONFIG="$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py"
LABELS="$BASE/labels/$SOURCE_TAG"
TRAIN_LOSS_CSV="$LABELS/train_quant_losses/train_losses_merged.csv"
VAL_LOSS_CSV="$LABELS/val_quant_losses/val_losses_merged.csv"
TRAIN_SPLIT="$LABELS/train_quant_losses/proxy_train_tokens.txt"
VAL_SPLIT="$LABELS/train_quant_losses/proxy_val_tokens.txt"
OFFICIAL_VAL_SPLIT="$LABELS/val_quant_losses/official_val_tokens.txt"
LAMBDA_JSON="$LABELS/train_quant_losses/rd_lambdas_train_only.json"
TRAIN_BPP="$BASE/labels/nuscenes_train_gpcc_per_frame_per_rate.csv"
VAL_BPP="$BASE/labels/nuscenes_val_gpcc_per_frame_per_rate.csv"
PROXY_INIT="$BASE/experiments/nuscenes_rate_aware_$SOURCE_TAG/best.pth"
PREDICTION_ROOT="$BASE/experiments/nuscenes_rate_aware_$SOURCE_TAG/val_prediction_cache"
FIXED_CURVE_CSV="$BASE/experiments/nuscenes_rate_aware_${SOURCE_TAG}_ddp7/final_map_bpp/nuscenes_rate_aware_measured_gpcc_map_bpp.csv"
OUT="$BASE/experiments/nuscenes_rate_aware_$TAG"
CANDIDATE_PREDICTIONS="$OUT/candidate_predictions"
MAP_SELECTION_OUT="$OUT/map_checkpoint_selection"
FINAL_OUT="$OUT/final_map_bpp"
LOG="$BASE/${TAG}_pipeline.log"
STATUS="$BASE/${TAG}_status.txt"

mkdir -p "$OUT" "$CANDIDATE_PREDICTIONS" "$MAP_SELECTION_OUT" "$FINAL_OUT"
exec 9>"$BASE/.${TAG}.lock"
if ! flock -n 9; then
    echo "The normalized-loss mAP-selected router pipeline is already active."
    exit 0
fi
exec >>"$LOG" 2>&1
rm -f "$BASE/${TAG}_FAILED" "$BASE/${TAG}_ALL_DONE"

record() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}
fail() {
    local code=$?
    record "FAILED exit=$code command=$BASH_COMMAND"
    touch "$BASE/${TAG}_FAILED"
    exit "$code"
}
trap fail ERR

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

record "Stage 1/5: preflight normalized-loss training inputs and cached detections"
for path in "$CONFIG" "$TRAIN_LOSS_CSV" "$VAL_LOSS_CSV" "$TRAIN_SPLIT" \
    "$VAL_SPLIT" "$OFFICIAL_VAL_SPLIT" "$LAMBDA_JSON" "$TRAIN_BPP" \
    "$VAL_BPP" "$PROXY_INIT" "$FIXED_CURVE_CSV"; do
    [[ -s "$path" ]]
done
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq 6 ]]
"$PYTHON" -m py_compile \
    "$CODE/train_nuscenes_rate_aware_proxy_ddp.py" \
    "$CODE/export_nuscenes_router_candidates_multigpu.py" \
    "$CODE/select_nuscenes_router_by_map_bpp.py"
LAMBDAS=$("$PYTHON" -c "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

record "Stage 2/5: seven-GPU DDP with per-loss-head standardization and unchanged BPP weight 1.0"
if [[ ! -s "$OUT/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
        --master_addr=127.0.0.1 --master_port=29588 --nproc_per_node=7 \
        "$CODE/train_nuscenes_rate_aware_proxy_ddp.py" \
        --dataset-format nuscenes --points-dir "$DATA" \
        --train-loss-csv "$TRAIN_LOSS_CSV" --val-loss-csv "$TRAIN_LOSS_CSV" \
        --train-bpp-csv "$TRAIN_BPP" --val-bpp-csv "$TRAIN_BPP" \
        --train-split "$TRAIN_SPLIT" --val-split "$VAL_SPLIT" \
        --test-split "$OFFICIAL_VAL_SPLIT" --test-loss-csv "$VAL_LOSS_CSV" \
        --test-bpp-csv "$VAL_BPP" --init-checkpoint "$PROXY_INIT" \
        --out-dir "$OUT" --lambdas $LAMBDAS \
        --target-scale 1.0 --voxel-size 0.16 0.16 0.16 \
        --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0 \
        --max-voxels 50000 --feat-dim 256 --epochs 40 \
        --batch-size 2 --workers 2 --lr 2e-4 --weight-decay 5e-4 \
        --rate-weight 1.0 --rd-weight 0.0 --selection-temperature 1.0 \
        --jitter-std 0.005 --patience 10 --seed 20260825 \
        >"$OUT/router_train.log" 2>&1
fi
[[ -s "$OUT/TRAINING_COMPLETE.json" && -s "$OUT/candidate_init.pth" ]]
[[ $(find "$OUT/checkpoints" -type f -name 'epoch_*.pth' | wc -l) -ge 1 ]]

record "Stage 3/5: export every checkpoint's official-val route decisions on seven GPUs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29589 --nproc_per_node=7 \
    "$CODE/export_nuscenes_router_candidates_multigpu.py" \
    --training-dir "$OUT" --output-dir "$CANDIDATE_PREDICTIONS" \
    >"$OUT/candidate_prediction_export.log" 2>&1
EXPECTED_CANDIDATES=$((1 + $(find "$OUT/checkpoints" -type f -name 'epoch_*.pth' | wc -l)))
[[ $(find "$CANDIDATE_PREDICTIONS" -maxdepth 1 -type f -name '*.csv' | wc -l) -eq "$EXPECTED_CANDIDATES" ]]

record "Stage 4/5: select the checkpoint by official validation mAP-BPP curve-area gain"
if [[ ! -s "$MAP_SELECTION_OUT/MAP_SELECTION_COMPLETE.json" ]]; then
    "$PYTHON" -u "$CODE/select_nuscenes_router_by_map_bpp.py" \
        --config "$CONFIG" --prediction-root "$PREDICTION_ROOT" \
        --candidate-predictions-dir "$CANDIDATE_PREDICTIONS" \
        --training-dir "$OUT" --fixed-curve-csv "$FIXED_CURVE_CSV" \
        --output-dir "$MAP_SELECTION_OUT" --parallel-workers 12 \
        >"$OUT/map_checkpoint_selection.log" 2>&1
fi
[[ -s "$OUT/best_map_bpp.pth" ]]
[[ -s "$OUT/test_rate_aware_predictions_map_selected.csv" ]]

record "Stage 5/5: evaluate and draw the mAP-selected final measured-GPCC curve"
SELECTED_EPOCH=$("$PYTHON" -c "import torch; print(torch.load('$OUT/best_map_bpp.pth',map_location='cpu').get('epoch',0))")
if [[ ! -s "$FINAL_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.png" ]]; then
    "$PYTHON" -u "$CODE/evaluate_nuscenes_multisweep_rate_aware_map_bpp.py" \
        --config "$CONFIG" --prediction-root "$PREDICTION_ROOT" \
        --rate-aware-predictions-csv "$OUT/test_rate_aware_predictions_map_selected.csv" \
        --output-dir "$FINAL_OUT" --checkpoint-epoch "$SELECTED_EPOCH" \
        --parallel-workers 12 >"$OUT/final_evaluation.log" 2>&1
fi
[[ -s "$FINAL_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.csv" ]]
[[ -s "$FINAL_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.png" ]]
touch "$BASE/${TAG}_ALL_DONE"
record "ALL DONE: standardized loss heads, unchanged BPP weight, mAP-BPP-selected checkpoint, and final plot"
