#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/DATA/sm/RACO-LPCC
PYTHON=/home/sm/miniconda3/envs/SparsePCGC/bin/python
TORCHRUN=/home/sm/miniconda3/envs/SparsePCGC/bin/torchrun
TOOLS=$ROOT/OpenPCDet/tools
CODE=$ROOT/routing/lrproxy
ROUTER_EVAL_SCRIPT=$ROOT/scripts/curve_tools/evaluate_kitti_router_from_pkls.py
MERGE_CURVE_SCRIPT=$ROOT/scripts/curve_tools/merge_kitti_detection_router_curve.py
AGGREGATE_RATE_SCRIPT=$ROOT/scripts/curve_tools/aggregate_kitti_router_rates.py
OUT=$ROOT/experiment_results/gpcc_current_q_ones_scratch_q128dist100_lrproxy_router_20260901
TMP=/tmp/sm_storage/gpcc_current_q_ones_scratch_q128dist100_lrproxy_router_20260901
STATE=$OUT/state
LOGS=$OUT/logs

TRAIN_SPLIT=$ROOT/OpenPCDet/data/kitti_fov/ImageSets/train.txt
VAL_SPLIT=$ROOT/OpenPCDet/data/kitti/ImageSets/val.txt
POINTS=$ROOT/OpenPCDet/data/kitti_fov/training/velodyne
CFG=$ROOT/integrations/openpcdet/configs/kitti_models/pv_rcnn_fov_geometry.yaml
DET_CKPT=$TOOLS/ckpt/model_non_reflectance.pth
RESIDUAL_CODE=$ROOT/reno/current_q_ones_coordinate_v1_20260831
RESIDUAL_CKPT=$ROOT/reno/current_q_ones_coordinate_runs_scratch_q128dist100_lr1e4_5ep_20260901/best_train_loss.pth
RESTORE_SCRIPT=$RESIDUAL_CODE/gpcc_current_q_ones_coordinate_restore.py
LOSS_SCRIPT=$ROOT/scripts/label_generation/export_kitti_pvrcnn_residual_loss.py
MERGE_SCRIPT=$ROOT/scripts/label_generation/merge_kitti_pvrcnn_residual_loss_shards.py
PLOT_SCRIPT=$ROOT/scripts/curve_tools/plot_kitti_gpcc_residual_map_bpp_curves.py
TRAIN_SCRIPT=$ROOT/scripts/training/train_kitti_lrproxy_residual_router_ddp.py

TRAIN_DECODED=$TMP/train_decoded
VAL_DECODED=/tmp/sm_storage/gpcc_current_q_ones_scratch_q128dist100_20260901/decoded
TRAIN_BPP=$ROOT/experiment_results/kitti_detection_gpcc_training_bpp_6scales/gpcc_train_details.csv
VAL_BPP=$ROOT/experiment_results/kitti_detection_gpcc_val_bpp_6scales/gpcc_val_details.csv
TRAIN_PLAIN_LOSS=$ROOT/experiment_results/kitti_detection_loss_labels_6scales/train_detection_loss_sensitivity.csv
VAL_PLAIN_LOSS=$ROOT/experiment_results/kitti_detection_loss_labels_6scales/val_detection_loss_sensitivity.csv
INIT_LRproxy=$TOOLS/router_work_dirs/lrproxy_kitti_pvrcnn_alltrain_trainloss_fullbpp_ddp7_20260829/best.pth
ROUTER_DIR=$TOOLS/router_work_dirs/lrproxy_kitti_pvrcnn_residual_scratch_q128dist100_alltrain_20260901
LABEL_DIR=$OUT/labels
FIXED_LINKS=$OUT/fixed_result_links
AP_DIR=$OUT/pvrcnn
RATE_DIR=$AP_DIR/gpcc

FIXED_CURVE=$ROOT/experiment_results/gpcc_current_q_ones_scratch_q128dist100_20260901/comparison/scratch_q128dist100_map_bpp.csv
PLAIN_ROUTER=$ROOT/experiment_results/kitti_detection_lrproxy_pvrcnn_zero_shot_20260829/pvrcnn/router_gpcc_curve.csv
RES_PKL_ROOT=$ROOT/OpenPCDet/output/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_fov_geometry/scratch_q128dist100_20260901/eval/epoch_no_number/val/scratch_q128dist100_20260901
PLAIN_RATE5_PKL=$ROOT/OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/uniform_direct_loss_6level_parallel_scale_5/eval/epoch_no_number/val/default/scale_0.015625/result.pkl
QUANT_MAP='1/2048,1/2048;1/1024,1/1024;1/512,1/512;1/256,1/256;1/128,1/128;1/64,1/64'

mkdir -p "$OUT" "$TMP" "$STATE" "$LOGS" "$ROUTER_DIR" "$LABEL_DIR" "$AP_DIR" "$RATE_DIR"
exec 9>"$OUT/pipeline.lock"
flock -n 9

stage() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOGS/pipeline.log"
}

fail() {
  code=$?
  printf '{"status":"failed","stage":"%s","exit_code":%d}\n' "${STAGE:-unknown}" "$code" >"$STATE/FAILED.json"
  exit "$code"
}
trap fail ERR

export PYTHONPATH="$CODE:$TOOLS:$ROOT/OpenPCDet:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

for required in "$TRAIN_SPLIT" "$VAL_SPLIT" "$DET_CKPT" "$RESIDUAL_CKPT" "$TRAIN_BPP" "$VAL_BPP" "$TRAIN_PLAIN_LOSS" "$VAL_PLAIN_LOSS" "$INIT_LRproxy" "$FIXED_CURVE" "$PLAIN_ROUTER"; do
  test -s "$required"
done

STAGE=restore_train
stage '1/8 restore five residual scales for complete KITTI FOV train split'
if [[ ! -f "$STATE/RESTORE_TRAIN_COMPLETE" ]]; then
  mkdir -p "$TRAIN_DECODED" "$OUT/train_restore_manifests"
  cd "$RESIDUAL_CODE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29741 \
    "$RESTORE_SCRIPT" --model-root "$RESIDUAL_CODE" --checkpoint "$RESIDUAL_CKPT" \
    --split "$TRAIN_SPLIT" --points-dir "$POINTS" --decoded-dir "$TRAIN_DECODED" \
    --summary-dir "$OUT/train_restore_manifests" --rate-ids 0,1,2,3,4 \
    >"$LOGS/restore_train.log" 2>&1
  touch "$STATE/RESTORE_TRAIN_COMPLETE"
fi

run_loss_shards() {
  local dataset_split=$1
  local split_file=$2
  local decoded_dir=$3
  local shard_dir=$4
  local log_prefix=$5
  local plain_loss_csv=$6
  mkdir -p "$shard_dir"
  local pids=()
  for rank in 0 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=$rank "$PYTHON" -u "$LOSS_SCRIPT" \
      --cfg-file "$CFG" --ckpt "$DET_CKPT" --dataset-split "$dataset_split" \
      --split-file "$split_file" --decoded-dir "$decoded_dir" \
      --decoded-rate-ids 0,1,2,3,4 --plain-rate-id 5 --plain-q-step-mm 64 \
      --reuse-plain-loss-csv "$plain_loss_csv" \
      --output-csv "$shard_dir/shard_${rank}.csv" --shard-id "$rank" --num-shards 7 \
      >"$LOGS/${log_prefix}_${rank}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

STAGE=train_loss
stage '2/8 calculate six-scale PV-RCNN losses for all train frames'
if [[ ! -f "$STATE/TRAIN_LOSS_COMPLETE" ]]; then
  run_loss_shards train "$TRAIN_SPLIT" "$TRAIN_DECODED" "$OUT/train_loss_shards" train_loss "$TRAIN_PLAIN_LOSS"
  "$PYTHON" "$MERGE_SCRIPT" --split "$TRAIN_SPLIT" --shard-dir "$OUT/train_loss_shards" \
    --num-shards 7 --output "$OUT/train_detection_losses.csv" >"$LOGS/merge_train_loss.log" 2>&1
  touch "$STATE/TRAIN_LOSS_COMPLETE"
fi

STAGE=val_loss
stage '3/8 calculate six-scale PV-RCNN losses for all val frames'
if [[ ! -f "$STATE/VAL_LOSS_COMPLETE" ]]; then
  run_loss_shards val "$VAL_SPLIT" "$VAL_DECODED" "$OUT/val_loss_shards" val_loss "$VAL_PLAIN_LOSS"
  "$PYTHON" "$MERGE_SCRIPT" --split "$VAL_SPLIT" --shard-dir "$OUT/val_loss_shards" \
    --num-shards 7 --output "$OUT/val_detection_losses.csv" >"$LOGS/merge_val_loss.log" 2>&1
  touch "$STATE/VAL_LOSS_COMPLETE"
fi

STAGE=router_smoke
stage '4/8 LRproxy full-checkpoint initialization and one-epoch smoke test'
if [[ ! -f "$STATE/ROUTER_SMOKE_COMPLETE" ]]; then
  SMOKE=$OUT/router_smoke
  mkdir -p "$SMOKE"
  cd "$CODE"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$TRAIN_SCRIPT" \
    --points-dir "$POINTS" --loss-csv "$OUT/train_detection_losses.csv" --bpp-csv "$TRAIN_BPP" \
    --train-split "$TRAIN_SPLIT" --init-checkpoint "$INIT_LRproxy" \
    --output-dir "$SMOKE" --epochs 1 --batch-size 8 --workers 0 --max-train-frames 16 \
    >"$LOGS/router_smoke.log" 2>&1
  touch "$STATE/ROUTER_SMOKE_COMPLETE"
fi

STAGE=router_train
stage '5/8 train LRproxy on all 3712 train frames; select lowest train regression loss'
if [[ ! -f "$STATE/ROUTER_TRAIN_COMPLETE" ]]; then
  cd "$CODE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29742 \
    "$TRAIN_SCRIPT" --points-dir "$POINTS" --loss-csv "$OUT/train_detection_losses.csv" \
    --bpp-csv "$TRAIN_BPP" --train-split "$TRAIN_SPLIT" --init-checkpoint "$INIT_LRproxy" \
    --output-dir "$ROUTER_DIR" --epochs 60 --save-every 10 \
    --batch-size 16 --workers 2 --seed 20260901 >"$LOGS/router_train.log" 2>&1
  touch "$STATE/ROUTER_TRAIN_COMPLETE"
fi

STAGE=route_export
stage '6/8 export six analytical routing operating points on official val'
if [[ ! -f "$STATE/ROUTE_EXPORT_COMPLETE" ]]; then
  cd "$CODE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$TORCHRUN" --nproc_per_node=7 \
    --master_addr=127.0.0.1 --master_port=29743 \
    export_kitti_lrproxy_router_labels.py --checkpoint "$ROUTER_DIR/best.pth" \
    --points-dir "$POINTS" --split "$VAL_SPLIT" --output-dir "$LABEL_DIR" \
    --prefix lrproxy_residual --batch-size 16 --workers 2 \
    >"$LOGS/route_export.log" 2>&1
  touch "$STATE/ROUTE_EXPORT_COMPLETE"
fi

STAGE=route_eval
stage '7/8 combine cached per-scale detections and calculate official mAP plus measured BPP'
if [[ ! -f "$STATE/ROUTE_EVAL_COMPLETE" ]]; then
  names=(
    combo_0_fg_0.000488_bg_0.000488
    combo_1_fg_0.000977_bg_0.000977
    combo_2_fg_0.001953_bg_0.001953
    combo_3_fg_0.003906_bg_0.003906
    combo_4_fg_0.007812_bg_0.007812
    combo_5_fg_0.015625_bg_0.015625
  )
  for index in 0 1 2 3 4; do
    mkdir -p "$FIXED_LINKS/${names[$index]}"
    ln -sfn "$RES_PKL_ROOT/rate_${index}/result.pkl" "$FIXED_LINKS/${names[$index]}/result.pkl"
  done
  mkdir -p "$FIXED_LINKS/${names[5]}"
  ln -sfn "$PLAIN_RATE5_PKL" "$FIXED_LINKS/${names[5]}/result.pkl"
  cd "$TOOLS"
  "$PYTHON" "$ROUTER_EVAL_SCRIPT" --cfg_file "$CFG" \
    --eval_dir "$FIXED_LINKS" --quant_map "$QUANT_MAP" \
    --manifest "$LABEL_DIR/lrproxy_residual_manifest.json" \
    --out "$AP_DIR/router_ap.csv" --save_mixed_pkls_dir "$AP_DIR/mixed_pkls" \
    >"$LOGS/route_ap.log" 2>&1
  cd "$ROOT"
  "$PYTHON" "$AGGREGATE_RATE_SCRIPT" --details-csv "$VAL_BPP" \
    --split-file "$VAL_SPLIT" --manifest "$LABEL_DIR/lrproxy_residual_manifest.json" \
    --out-dir "$RATE_DIR" >"$LOGS/route_bpp.log" 2>&1
  "$PYTHON" "$MERGE_CURVE_SCRIPT" --ap_csv "$AP_DIR/router_ap.csv" \
    --gpcc_csv "$RATE_DIR/router_average_results.csv" --out "$AP_DIR/router_gpcc_residual_curve.csv" \
    >"$LOGS/merge_router_curve.log" 2>&1
  touch "$STATE/ROUTE_EVAL_COMPLETE"
fi

STAGE=plot
stage '8/8 plot four fixed/routed curves with and without residual restoration'
"$PYTHON" "$PLOT_SCRIPT" --fixed-csv "$FIXED_CURVE" --plain-router-csv "$PLAIN_ROUTER" \
  --residual-router-csv "$AP_DIR/router_gpcc_residual_curve.csv" --output-dir "$OUT/comparison" \
  >"$LOGS/plot.log" 2>&1

printf '{"status":"complete","residual_checkpoint":"%s","router_checkpoint":"%s","selection":"lowest complete-train regression loss","official_val_used_for_selection":false}\n' \
  "$RESIDUAL_CKPT" "$ROUTER_DIR/best.pth" >"$OUT/PIPELINE_COMPLETE.json"
touch "$STATE/COMPLETE"
stage 'pipeline complete'
