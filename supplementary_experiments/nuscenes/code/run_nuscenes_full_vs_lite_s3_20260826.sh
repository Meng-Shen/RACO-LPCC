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
SOURCE=multisweep_cbgs_bestmap_epoch4_router_7gpu_20260825
LABELS="$BASE/labels/$SOURCE"
TRAIN_LOSS="$LABELS/train_quant_losses/train_losses_merged.csv"
VAL_LOSS="$LABELS/val_quant_losses/val_losses_merged.csv"
TRAIN_SPLIT="$LABELS/train_quant_losses/proxy_train_tokens.txt"
VAL_SPLIT="$LABELS/train_quant_losses/proxy_val_tokens.txt"
TEST_SPLIT="$LABELS/val_quant_losses/official_val_tokens.txt"
TRAIN_BPP="$BASE/labels/nuscenes_train_gpcc_per_frame_per_rate.csv"
VAL_BPP="$BASE/labels/nuscenes_val_gpcc_per_frame_per_rate.csv"
LAMBDA_JSON="$LABELS/train_quant_losses/rd_lambdas_train_only.json"
LEGACY_INIT="$BASE/experiments/nuscenes_rate_aware_multisweep_cbgs_bestmap_epoch4_normloss_mapselect_7gpu_20260825/best_map_bpp.pth"
PREDICTION_ROOT="$BASE/experiments/nuscenes_rate_aware_$SOURCE/val_prediction_cache"
FIXED_CURVE="$BASE/experiments/nuscenes_rate_aware_${SOURCE}_ddp7/final_map_bpp/nuscenes_rate_aware_measured_gpcc_map_bpp.csv"
FULL="$BASE/experiments/nuscenes_sixloss_monotonic_full_20260826"
LITE="$BASE/experiments/nuscenes_sixloss_monotonic_lite_s3_20260826"
SMOKE_FULL="$BASE/experiments/nuscenes_sixloss_monotonic_full_smoke_20260826"
SMOKE_LITE="$BASE/experiments/nuscenes_sixloss_monotonic_lite_s3_smoke_20260826"
COMPARE="$BASE/experiments/nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826"
LOG="$BASE/nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826_pipeline.log"
STATUS="$BASE/nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826_status.txt"

mkdir -p "$FULL" "$LITE" "$SMOKE_FULL" "$SMOKE_LITE" "$COMPARE"
exec 9>"$BASE/.nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826.lock"
if ! flock -n 9; then
    echo "The nuScenes full-vs-Lite-S3 pipeline is already active."
    exit 0
fi
exec >>"$LOG" 2>&1

record() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}
fail() {
    local code=$?
    record "FAILED exit=$code command=$BASH_COMMAND"
    touch "$BASE/nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826_FAILED"
    exit "$code"
}
trap fail ERR

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
LAMBDAS=$($PYTHON -c "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

COMMON_ARGS=(
    --points-dir "$DATA"
    --train-loss-csv "$TRAIN_LOSS" --val-loss-csv "$TRAIN_LOSS"
    --train-bpp-csv "$TRAIN_BPP" --val-bpp-csv "$TRAIN_BPP"
    --train-split "$TRAIN_SPLIT" --val-split "$VAL_SPLIT"
    --test-split "$TEST_SPLIT" --test-loss-csv "$VAL_LOSS" --test-bpp-csv "$VAL_BPP"
    --lambdas $LAMBDAS
    --voxel-size 0.16 0.16 0.16
    --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0
    --max-voxels 50000 --feat-dim 256 --batch-size 2 --workers 2
    --backbone-lr 5e-5 --head-lr 5e-4 --weight-decay 5e-4
    --loss-weight 2.0 --rate-weight 1.0 --jitter-std 0.005 --seed 20260826
)

record "Stage 1/9: preflight assets and Python syntax"
for path in "$TRAIN_LOSS" "$VAL_LOSS" "$TRAIN_SPLIT" "$VAL_SPLIT" "$TEST_SPLIT" \
    "$TRAIN_BPP" "$VAL_BPP" "$LAMBDA_JSON" "$LEGACY_INIT" "$FIXED_CURVE"; do
    [[ -s "$path" ]]
done
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq 6 ]]
$PYTHON -m py_compile \
    "$CODE/absolute_loss_monotonic_rate_proxy.py" \
    "$CODE/lite_s3_absolute_loss_monotonic_rate_proxy.py" \
    "$CODE/gpu_voxelizer.py" \
    "$CODE/train_nuscenes_sixloss_monotonic_router_ddp.py" \
    "$CODE/export_nuscenes_sixloss_router_candidates_multigpu.py" \
    "$CODE/plot_nuscenes_full_vs_lite_s3.py"

record "Stage 2/9: full-router one-batch smoke test"
if [[ ! -s "$SMOKE_FULL/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u "$CODE/train_nuscenes_sixloss_monotonic_router_ddp.py" \
        "${COMMON_ARGS[@]}" --model-variant full --init-kind legacy \
        --init-checkpoint "$LEGACY_INIT" --out-dir "$SMOKE_FULL" \
        --epochs 1 --max-train-frames 4 --max-val-frames 4 \
        >"$SMOKE_FULL/smoke.log" 2>&1
fi
grep -q 'bpp_monotonic=True' "$SMOKE_FULL/smoke.log"

record "Stage 3/9: train full six-loss monotonic router on seven GPUs"
if [[ ! -s "$FULL/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $PYTHON -m torch.distributed.run \
        --master_addr=127.0.0.1 --master_port=29626 --nproc_per_node=7 \
        "$CODE/train_nuscenes_sixloss_monotonic_router_ddp.py" \
        "${COMMON_ARGS[@]}" --model-variant full --init-kind legacy \
        --init-checkpoint "$LEGACY_INIT" --out-dir "$FULL" --epochs 8 \
        >"$FULL/train.log" 2>&1
fi
[[ $(find "$FULL/checkpoints" -type f -name 'epoch_*.pth' | wc -l) -eq 8 ]]

record "Stage 4/9: export and officially select full-router checkpoint by validation mAP-BPP AUC"
mkdir -p "$FULL/candidate_predictions" "$FULL/map_checkpoint_selection"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $PYTHON -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29627 --nproc_per_node=7 \
    "$CODE/export_nuscenes_sixloss_router_candidates_multigpu.py" \
    --training-dir "$FULL" --output-dir "$FULL/candidate_predictions" \
    >"$FULL/export.log" 2>&1
if [[ ! -s "$FULL/map_checkpoint_selection/MAP_SELECTION_COMPLETE.json" ]]; then
    $PYTHON -u "$CODE/select_nuscenes_router_by_map_bpp.py" \
        --config "$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py" \
        --prediction-root "$PREDICTION_ROOT" \
        --candidate-predictions-dir "$FULL/candidate_predictions" \
        --training-dir "$FULL" --fixed-curve-csv "$FIXED_CURVE" \
        --output-dir "$FULL/map_checkpoint_selection" --parallel-workers 12 \
        >"$FULL/map_selection.log" 2>&1
fi
[[ -s "$FULL/best_map_bpp.pth" ]]

record "Stage 5/9: Lite-S3 smoke test including full-checkpoint transfer"
if [[ ! -s "$SMOKE_LITE/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u "$CODE/train_nuscenes_sixloss_monotonic_router_ddp.py" \
        "${COMMON_ARGS[@]}" --model-variant lite_s3 --init-kind full_sixloss \
        --init-checkpoint "$FULL/best_map_bpp.pth" --out-dir "$SMOKE_LITE" \
        --epochs 1 --max-train-frames 4 --max-val-frames 4 \
        >"$SMOKE_LITE/smoke.log" 2>&1
fi
grep -q 'bpp_monotonic=True' "$SMOKE_LITE/smoke.log"

record "Stage 6/9: train Lite-S3 router on seven GPUs from selected full router"
if [[ ! -s "$LITE/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $PYTHON -m torch.distributed.run \
        --master_addr=127.0.0.1 --master_port=29628 --nproc_per_node=7 \
        "$CODE/train_nuscenes_sixloss_monotonic_router_ddp.py" \
        "${COMMON_ARGS[@]}" --model-variant lite_s3 --init-kind full_sixloss \
        --init-checkpoint "$FULL/best_map_bpp.pth" --out-dir "$LITE" --epochs 8 \
        >"$LITE/train.log" 2>&1
fi
[[ $(find "$LITE/checkpoints" -type f -name 'epoch_*.pth' | wc -l) -eq 8 ]]

record "Stage 7/9: export Lite-S3 candidates"
mkdir -p "$LITE/candidate_predictions" "$LITE/map_checkpoint_selection"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $PYTHON -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29629 --nproc_per_node=7 \
    "$CODE/export_nuscenes_sixloss_router_candidates_multigpu.py" \
    --training-dir "$LITE" --output-dir "$LITE/candidate_predictions" \
    >"$LITE/export.log" 2>&1

record "Stage 8/9: officially select Lite-S3 checkpoint by validation mAP-BPP AUC"
if [[ ! -s "$LITE/map_checkpoint_selection/MAP_SELECTION_COMPLETE.json" ]]; then
    $PYTHON -u "$CODE/select_nuscenes_router_by_map_bpp.py" \
        --config "$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py" \
        --prediction-root "$PREDICTION_ROOT" \
        --candidate-predictions-dir "$LITE/candidate_predictions" \
        --training-dir "$LITE" --fixed-curve-csv "$FIXED_CURVE" \
        --output-dir "$LITE/map_checkpoint_selection" --parallel-workers 12 \
        >"$LITE/map_selection.log" 2>&1
fi
[[ -s "$LITE/best_map_bpp.pth" ]]

record "Stage 9/9: draw linear-BPP full-vs-Lite-S3 comparison"
$PYTHON -u "$CODE/plot_nuscenes_full_vs_lite_s3.py" \
    --fixed-curve-csv "$FIXED_CURVE" \
    --full-selection-dir "$FULL/map_checkpoint_selection" \
    --lite-selection-dir "$LITE/map_checkpoint_selection" \
    --output-dir "$COMPARE" >"$COMPARE/plot.log" 2>&1
touch "$BASE/nuscenes_sixloss_monotonic_full_vs_lite_s3_20260826_ALL_DONE"
record "ALL DONE: fixed G-PCC, full six-loss router, and Lite-S3 comparison are ready"
