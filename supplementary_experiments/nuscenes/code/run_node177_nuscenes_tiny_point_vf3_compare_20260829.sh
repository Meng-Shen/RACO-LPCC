#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${RACO_NUSCENES_ROOT:-/home/sm/raco_rate_aware_nuscenes_20260822}"
ENV=/home/sm/miniconda3/envs/openmmlab
PYTHON=$ENV/bin/python
MIM=$ENV/lib/python3.8/site-packages/mmdet3d/.mim
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR"
CONFIG_ROOT="$SCRIPT_DIR/../configs"
DATA=$BASE/data/nuscenes
SOURCE=multisweep_cbgs_bestmap_epoch4_router_7gpu_20260825
LABELS=$BASE/labels/$SOURCE
TRAIN_LOSS=$LABELS/train_quant_losses/train_losses_merged.csv
VAL_LOSS=$LABELS/val_quant_losses/val_losses_merged.csv
TRAIN_SPLIT=$LABELS/train_quant_losses/proxy_train_tokens.txt
VAL_SPLIT=$LABELS/train_quant_losses/proxy_val_tokens.txt
TEST_SPLIT=$LABELS/val_quant_losses/official_val_tokens.txt
TRAIN_BPP=$BASE/labels/nuscenes_train_gpcc_per_frame_per_rate.csv
VAL_BPP=$BASE/labels/nuscenes_val_gpcc_per_frame_per_rate.csv
LAMBDA_JSON=$LABELS/train_quant_losses/rd_lambdas_train_only.json
INIT=$BASE/experiments/nuscenes_tiny_point_vf7_resume_e20_to_e60_alltrain_trainloss_20260828/best.pth
PREDICTION_ROOT=$BASE/experiments/nuscenes_rate_aware_$SOURCE/val_prediction_cache
FIXED_CURVE=$BASE/experiments/nuscenes_rate_aware_${SOURCE}_ddp7/final_map_bpp/nuscenes_rate_aware_measured_gpcc_map_bpp.csv
VF7_CURVE=$BASE/experiments/nuscenes_tiny_point_vf7_resume_e20_to_e60_vs_gpcc_20260828/nuscenes_tiny_point_vf7_vs_gpcc_map_bpp.csv
SMOKE_OUT=$BASE/experiments/nuscenes_tiny_point_vf3_smoke_20260829
TRAIN_OUT=$BASE/experiments/nuscenes_tiny_point_vf3_alltrain_trainloss_ddp7_20260829
COMPARE_OUT=$BASE/experiments/nuscenes_gpcc_vf7_vf3_20260829
STATE=$TRAIN_OUT/state
LOG=$BASE/nuscenes_tiny_point_vf3_20260829_pipeline.log

mkdir -p "$SMOKE_OUT" "$TRAIN_OUT" "$COMPARE_OUT" "$STATE"
exec 9>"$TRAIN_OUT/pipeline.lock"
if ! flock -n 9; then
  echo "The node-177 nuScenes VF3 pipeline is already active."
  exit 2
fi
exec >>"$LOG" 2>&1

record() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() {
  code=$?
  printf '{"status":"failed","exit_code":%d,"stage":"%s"}\n' "$code" "${STAGE:-unknown}" >"$STATE/FAILED.json"
  record "FAILED stage=${STAGE:-unknown} exit=$code command=$BASH_COMMAND"
  exit "$code"
}
trap fail ERR

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
LAMBDAS=$($PYTHON -c "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

for path in "$TRAIN_LOSS" "$VAL_LOSS" "$TRAIN_SPLIT" "$VAL_SPLIT" "$TEST_SPLIT" \
  "$TRAIN_BPP" "$VAL_BPP" "$LAMBDA_JSON" "$INIT" "$FIXED_CURVE" "$VF7_CURVE"; do [[ -s "$path" ]]; done
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq 6 ]]

COMMON=(
  --points-dir "$DATA" --train-loss-csv "$TRAIN_LOSS" --train-bpp-csv "$TRAIN_BPP"
  --train-split "$TRAIN_SPLIT" "$VAL_SPLIT"
  --test-split "$TEST_SPLIT" --test-loss-csv "$VAL_LOSS" --test-bpp-csv "$VAL_BPP"
  --model-variant tiny_point_vf3 --init-kind tiny_point_full --init-checkpoint "$INIT"
  --lambdas $LAMBDAS --voxel-size 0.16 0.16 0.16
  --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0
  --max-voxels 50000 --feat-dim 256 --workers 2
  --backbone-lr 5e-4 --head-lr 2.5e-4 --weight-decay 5e-4
  --loss-weight 2.0 --rate-weight 1.0 --jitter-std 0.005 --seed 20260829
)

STAGE=smoke
record "nuScenes 1/5 VF7-to-VF3 single-GPU smoke test"
if [[ ! -s "$SMOKE_OUT/TRAINING_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$CODE/train_nuscenes_tiny_point_vf3_router_ddp.py" \
    "${COMMON[@]}" --out-dir "$SMOKE_OUT" --epochs 1 --patience 2 \
    --batch-size 4 --workers 0 --max-train-frames 8 \
    >"$SMOKE_OUT/train.log" 2>&1
fi
"$PYTHON" -c "import json; p=json.load(open('$SMOKE_OUT/initialization_report.json')); s=json.load(open('$SMOKE_OUT/TRAINING_COMPLETE.json')); assert p['mapped_source_channels']==[4,5,6]; assert not p['new_backbone_randomly_initialized']; assert s['model_alias']=='TinyPoint-VF3'"
touch "$STATE/SMOKE_COMPLETE"

STAGE=train
record "nuScenes 2/5 seven-GPU full-train regression-loss selection"
if [[ ! -s "$TRAIN_OUT/TRAINING_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29741 --nproc_per_node=7 \
    "$CODE/train_nuscenes_tiny_point_vf3_router_ddp.py" \
    "${COMMON[@]}" --out-dir "$TRAIN_OUT" --epochs 40 --patience 10 --batch-size 8 \
    >"$TRAIN_OUT/train.log" 2>&1
fi
[[ -s "$TRAIN_OUT/checkpoints/epoch_best.pth" ]]
touch "$STATE/TRAIN_COMPLETE"

STAGE=export
record "nuScenes 3/5 export selected-checkpoint routing labels"
mkdir -p "$TRAIN_OUT/candidate_predictions" "$TRAIN_OUT/map_checkpoint_selection"
if [[ ! -s "$STATE/EXPORT_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29742 --nproc_per_node=7 \
    "$CODE/export_nuscenes_tiny_point_vf3_candidates_multigpu.py" \
    --training-dir "$TRAIN_OUT" --output-dir "$TRAIN_OUT/candidate_predictions" \
    >"$TRAIN_OUT/export.log" 2>&1
fi
touch "$STATE/EXPORT_COMPLETE"

STAGE=official_eval
record "nuScenes 4/5 official CenterPoint mAP evaluation"
if [[ ! -s "$TRAIN_OUT/map_checkpoint_selection/MAP_SELECTION_COMPLETE.json" ]]; then
  "$PYTHON" -u "$CODE/select_nuscenes_router_by_map_bpp.py" \
    --config "$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py" \
    --prediction-root "$PREDICTION_ROOT" \
    --candidate-predictions-dir "$TRAIN_OUT/candidate_predictions" \
    --training-dir "$TRAIN_OUT" --fixed-curve-csv "$FIXED_CURVE" \
    --output-dir "$TRAIN_OUT/map_checkpoint_selection" --parallel-workers 6 \
    --single-checkpoint-evaluation >"$TRAIN_OUT/map_selection.log" 2>&1
fi
touch "$STATE/EVAL_COMPLETE"

STAGE=plot
record "nuScenes 5/5 fixed G-PCC / VF7 / VF3 comparison"
"$PYTHON" -u "$CODE/plot_vf7_vf3_comparison.py" --task nuscenes \
  --baseline "$FIXED_CURVE" --vf7 "$VF7_CURVE" \
  --vf3 "$TRAIN_OUT/map_checkpoint_selection" --output-dir "$COMPARE_OUT" \
  >"$COMPARE_OUT/plot.log" 2>&1
touch "$STATE/PLOT_COMPLETE"

STAGE=complete
printf '{"status":"complete","task":"nuScenes CenterPoint","model_alias":"TinyPoint-VF3","input_feature_dim":3,"comparison":"fixed G-PCC vs VF7 vs VF3"}\n' >"$TRAIN_OUT/PIPELINE_COMPLETE.json"
touch "$STATE/COMPLETE"
record "node-177 nuScenes VF3 pipeline complete"
