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
INIT="$BASE/experiments/nuscenes_tiny_point_vf7_alltrain_trainloss_20260828/best.pth"
PREDICTION_ROOT="$BASE/experiments/nuscenes_rate_aware_$SOURCE/val_prediction_cache"
FIXED_CURVE="$BASE/experiments/nuscenes_rate_aware_${SOURCE}_ddp7/final_map_bpp/nuscenes_rate_aware_measured_gpcc_map_bpp.csv"
TRAIN_OUT="$BASE/experiments/nuscenes_tiny_point_vf7_resume_e20_to_e60_alltrain_trainloss_20260828"
PLOT_OUT="$BASE/experiments/nuscenes_tiny_point_vf7_resume_e20_to_e60_vs_gpcc_20260828"
STATE="$TRAIN_OUT/state"
LOG="$BASE/nuscenes_tiny_point_vf7_resume_e20_to_e60_20260828_pipeline.log"
LOCK="$BASE/.nuscenes_tiny_point_vf7_resume_e20_to_e60_20260828.lock"
PID_FILE="$BASE/.nuscenes_tiny_point_vf7_resume_e20_to_e60_20260828.pid"

mkdir -p "$TRAIN_OUT" "$PLOT_OUT" "$STATE"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "The nuScenes TinyPoint-VF7 continuation pipeline is already active."
  exit 2
fi
echo $$ >"$PID_FILE"
exec >>"$LOG" 2>&1

record() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}
fail() {
  code=$?
  printf '{"status":"failed","exit_code":%d,"command":"%s"}\n' "$code" "$BASH_COMMAND" >"$STATE/FAILED.json"
  record "FAILED exit=$code command=$BASH_COMMAND"
  exit "$code"
}
trap fail ERR

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
LAMBDAS=$($PYTHON -c "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

for path in "$TRAIN_LOSS" "$VAL_LOSS" "$TRAIN_SPLIT" "$VAL_SPLIT" "$TEST_SPLIT" \
  "$TRAIN_BPP" "$VAL_BPP" "$LAMBDA_JSON" "$INIT" "$FIXED_CURVE"; do
  [[ -s "$path" ]]
done
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq 6 ]]

cat >"$TRAIN_OUT/RESUME_METADATA.json" <<EOF
{
  "status": "launched",
  "source_checkpoint": "$INIT",
  "source_global_epoch": 20,
  "additional_max_epochs": 40,
  "effective_max_global_epoch": 60,
  "optimizer_policy": "warm restart because the source cosine schedule reached zero learning rate",
  "backbone_lr": 0.0005,
  "head_lr": 0.00025,
  "early_stopping": "minimum full-training total loss, patience 10",
  "training_split": "all 28130 merged training frames; no holdout",
  "test_used_for_training_or_selection": false
}
EOF

record "Stage 1/4 seven-GPU warm-restart continuation from global epoch 20"
if [[ ! -s "$TRAIN_OUT/TRAINING_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29680 --nproc_per_node=7 \
    "$CODE/train_nuscenes_tiny_point_vf7_router_ddp.py" \
    --points-dir "$DATA" \
    --train-loss-csv "$TRAIN_LOSS" --train-bpp-csv "$TRAIN_BPP" \
    --train-split "$TRAIN_SPLIT" "$VAL_SPLIT" \
    --test-split "$TEST_SPLIT" --test-loss-csv "$VAL_LOSS" --test-bpp-csv "$VAL_BPP" \
    --out-dir "$TRAIN_OUT" --model-variant tiny_point_vf7 \
    --init-kind tiny_point_full --init-checkpoint "$INIT" \
    --lambdas $LAMBDAS \
    --voxel-size 0.16 0.16 0.16 \
    --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0 \
    --max-voxels 50000 --feat-dim 256 --workers 2 \
    --epochs 40 --patience 10 --batch-size 8 \
    --backbone-lr 5e-4 --head-lr 2.5e-4 --weight-decay 5e-4 \
    --loss-weight 2.0 --rate-weight 1.0 --jitter-std 0.005 --seed 20260828 \
    >"$TRAIN_OUT/train.log" 2>&1
fi
[[ -s "$TRAIN_OUT/checkpoints/epoch_best.pth" ]]
touch "$STATE/TRAIN_COMPLETE"

record "Stage 2/4 export the training-loss-selected checkpoint"
mkdir -p "$TRAIN_OUT/candidate_predictions" "$TRAIN_OUT/map_checkpoint_selection"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29681 --nproc_per_node=7 \
  "$CODE/export_nuscenes_tiny_point_vf7_candidates_multigpu.py" \
  --training-dir "$TRAIN_OUT" --output-dir "$TRAIN_OUT/candidate_predictions" \
  >"$TRAIN_OUT/export.log" 2>&1
touch "$STATE/EXPORT_COMPLETE"

record "Stage 3/4 official nuScenes validation of the single selected checkpoint"
"$PYTHON" -u "$CODE/select_nuscenes_router_by_map_bpp.py" \
  --config "$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py" \
  --prediction-root "$PREDICTION_ROOT" \
  --candidate-predictions-dir "$TRAIN_OUT/candidate_predictions" \
  --training-dir "$TRAIN_OUT" --fixed-curve-csv "$FIXED_CURVE" \
  --output-dir "$TRAIN_OUT/map_checkpoint_selection" --parallel-workers 6 \
  --single-checkpoint-evaluation >"$TRAIN_OUT/map_selection.log" 2>&1
[[ -s "$TRAIN_OUT/best_map_bpp.pth" ]]
touch "$STATE/SELECTION_COMPLETE"

record "Stage 4/4 draw total mAP-BPP plot"
"$PYTHON" -u "$CODE/plot_nuscenes_tiny_point_vf7_vs_gpcc.py" \
  --fixed-curve-csv "$FIXED_CURVE" \
  --tiny-selection-dir "$TRAIN_OUT/map_checkpoint_selection" \
  --output-dir "$PLOT_OUT" >"$PLOT_OUT/plot.log" 2>&1
touch "$STATE/PLOT_COMPLETE"
printf '{"status":"complete","model_alias":"TinyPoint-VF7","task":"nuScenes CenterPoint detection","source_global_epoch":20,"effective_max_global_epoch":60}\n' >"$TRAIN_OUT/PIPELINE_COMPLETE.json"
touch "$STATE/COMPLETE"
record "TinyPoint-VF7 nuScenes continuation pipeline complete"
