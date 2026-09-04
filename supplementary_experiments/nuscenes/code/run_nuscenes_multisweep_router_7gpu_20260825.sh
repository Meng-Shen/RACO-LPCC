#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${RACO_NUSCENES_ROOT:-/home/sm/raco_rate_aware_nuscenes_20260822}"
ENV=/home/sm/miniconda3/envs/openmmlab
PYTHON="$ENV/bin/python"
MIM="$ENV/lib/python3.8/site-packages/mmdet3d/.mim"
DATA="$BASE/data/nuscenes"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR"
CONFIG_ROOT="$SCRIPT_DIR/../configs"
TAG=multisweep_cbgs_bestmap_epoch4_router_7gpu_20260825
CONFIG="$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_20260824_nus-3d.py"
DETECTOR_CKPT="$BASE/experiments/centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_5e_20260824/best_NuScenes metric_pred_instances_3d_NuScenes_mAP_epoch_4.pth"
EXPORTER="$CODE/export_nuscenes_centerpoint_quant_multisweep.py"
EVALUATOR="$CODE/evaluate_nuscenes_multisweep_rate_aware_map_bpp.py"
TRAIN_BPP="$BASE/labels/nuscenes_train_gpcc_per_frame_per_rate.csv"
VAL_BPP="$BASE/labels/nuscenes_val_gpcc_per_frame_per_rate.csv"
RUN_LABELS="$BASE/labels/$TAG"
TRAIN_LOSS_ROOT="$RUN_LABELS/train_quant_losses"
VAL_LOSS_ROOT="$RUN_LABELS/val_quant_losses"
TRAIN_LOSS_CSV="$TRAIN_LOSS_ROOT/train_losses_merged.csv"
VAL_LOSS_CSV="$VAL_LOSS_ROOT/val_losses_merged.csv"
PROXY_TRAIN_SPLIT="$TRAIN_LOSS_ROOT/proxy_train_tokens.txt"
PROXY_VAL_SPLIT="$TRAIN_LOSS_ROOT/proxy_val_tokens.txt"
OFFICIAL_VAL_SPLIT="$VAL_LOSS_ROOT/official_val_tokens.txt"
LAMBDA_JSON="$TRAIN_LOSS_ROOT/rd_lambdas_train_only.json"
WARM_PROXY_OUT="$BASE/experiments/nuscenes_rate_aware_$TAG"
PROXY_INIT="$WARM_PROXY_OUT/best.pth"
RATE_PROXY_OUT="${WARM_PROXY_OUT}_ddp7"
# The detector prediction cache is already complete and is independent of the
# route-network checkpoint, so retain it instead of recomputing 36,114 passes.
PREDICTION_ROOT="$WARM_PROXY_OUT/val_prediction_cache"
MAP_OUT="$RATE_PROXY_OUT/final_map_bpp"
QSTEPS=2048,1024,512,256,128,64
# Two independent exporters per GPU fill the otherwise CPU/quantization-bound
# pipeline while staying within the 32 logical CPU cores and 251 GiB RAM.
NSHARDS=14
LOG="$BASE/${TAG}_pipeline.log"
STATUS="$BASE/${TAG}_status.txt"

mkdir -p "$TRAIN_LOSS_ROOT" "$VAL_LOSS_ROOT" "$RATE_PROXY_OUT" \
    "$PREDICTION_ROOT" "$MAP_OUT"
exec 9>"$BASE/.${TAG}.lock"
if ! flock -n 9; then
    echo "The seven-GPU multi-sweep routing pipeline is already active."
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

record "Stage 1/5: preflight full multi-sweep detector, measured BPP, and proxy initialization"
[[ -s "$CONFIG" && -s "$DETECTOR_CKPT" && -s "$PROXY_INIT" ]]
[[ -s "$EXPORTER" && -s "$EVALUATOR" ]]
[[ -s "$TRAIN_BPP" && -s "$VAL_BPP" ]]
[[ $(wc -l <"$TRAIN_BPP") -eq 168781 ]]
[[ $(wc -l <"$VAL_BPP") -eq 36115 ]]
"$PYTHON" -c "from mmengine.config import Config; c=Config.fromfile('$CONFIG'); s=[x for x in c.train_pipeline if x.type=='LoadPointsFromMultiSweeps']; assert len(s)==1 and s[0].sweeps_num==9 and list(s[0].use_dim)==[0,1,2]; assert c.model.pts_voxel_encoder.num_features==3; print('preflight: keyframe + 9 sweeps, XYZ only')"
"$PYTHON" -c "import torch; x=torch.load('$PROXY_INIT',map_location='cpu'); assert x.get('model_type')=='five_loss_heads_plus_one_six_rate_bpp_head'; print('preflight: nuScenes-adapted five-loss+BPP proxy initialization')"

record "Stage 2/5: export 14 deterministic loss shards, two per GPU on GPUs 0-6"
loss_pids=()
for ((shard=0; shard<NSHARDS; shard++)); do
    gpu=$((shard % 7))
    train_dir="$TRAIN_LOSS_ROOT/shard_$shard"
    val_dir="$VAL_LOSS_ROOT/shard_$shard"
    mkdir -p "$train_dir" "$val_dir"
    if [[ ! -s "$train_dir/loss.manifest.json" || ! -s "$val_dir/loss.manifest.json" ]]; then
        (
            if [[ ! -s "$train_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$EXPORTER" \
                    --mode loss --config "$CONFIG" --checkpoint "$DETECTOR_CKPT" \
                    --data-root "$DATA" --split train --qsteps-mm "$QSTEPS" \
                    --output "$train_dir/loss.csv" --shard-id "$shard" \
                    --num-shards "$NSHARDS" --device cuda:0 --log-every 20 \
                    >"$train_dir/export.log" 2>&1
            fi
            if [[ ! -s "$val_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$EXPORTER" \
                    --mode loss --config "$CONFIG" --checkpoint "$DETECTOR_CKPT" \
                    --data-root "$DATA" --split val --qsteps-mm "$QSTEPS" \
                    --output "$val_dir/loss.csv" --shard-id "$shard" \
                    --num-shards "$NSHARDS" --device cuda:0 --log-every 20 \
                    >"$val_dir/export.log" 2>&1
            fi
        ) &
        loss_pids+=("$!")
    fi
done
loss_failed=0
for pid in "${loss_pids[@]}"; do
    wait "$pid" || loss_failed=1
done
[[ "$loss_failed" -eq 0 ]]
[[ $(find "$TRAIN_LOSS_ROOT" -type f -name loss.manifest.json | wc -l) -eq "$NSHARDS" ]]
[[ $(find "$VAL_LOSS_ROOT" -type f -name loss.manifest.json | wc -l) -eq "$NSHARDS" ]]

record "Stage 3/5: merge loss labels, split train-only proxy data, and calibrate six lambdas"
if [[ ! -s "$TRAIN_LOSS_CSV" ]]; then
    "$PYTHON" "$CODE/merge_nuscenes_quant_losses.py" \
        --shard-root "$TRAIN_LOSS_ROOT" --output-csv "$TRAIN_LOSS_CSV" \
        --train-split "$PROXY_TRAIN_SPLIT" --val-split "$PROXY_VAL_SPLIT" \
        --val-percent 10
fi
if [[ ! -s "$VAL_LOSS_CSV" ]]; then
    "$PYTHON" "$CODE/merge_nuscenes_loss_shards.py" \
        --shard-root "$VAL_LOSS_ROOT" --output-csv "$VAL_LOSS_CSV" \
        --expected-samples 6019 --tokens-out "$OFFICIAL_VAL_SPLIT"
fi
if [[ ! -s "$LAMBDA_JSON" ]]; then
    "$PYTHON" "$CODE/select_scannet_rd_lambdas.py" \
        --dataset-format nuscenes --loss-csv "$TRAIN_LOSS_CSV" \
        --bpp-csv "$TRAIN_BPP" --split-file "$PROXY_TRAIN_SPLIT" \
        --output-json "$LAMBDA_JSON"
fi
LAMBDAS=$("$PYTHON" -c "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

record "Stage 4/5: train the loss+BPP router with seven-GPU DDP from the current best checkpoint"
PRED_SHARDS=6
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq "$PRED_SHARDS" ]]
if [[ ! -s "$RATE_PROXY_OUT/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PYTHON" -m torch.distributed.run \
        --master_addr=127.0.0.1 --master_port=29587 --nproc_per_node=7 \
        "$CODE/train_nuscenes_rate_aware_proxy_ddp.py" \
        --dataset-format nuscenes --points-dir "$DATA" \
        --train-loss-csv "$TRAIN_LOSS_CSV" --val-loss-csv "$TRAIN_LOSS_CSV" \
        --train-bpp-csv "$TRAIN_BPP" --val-bpp-csv "$TRAIN_BPP" \
        --train-split "$PROXY_TRAIN_SPLIT" --val-split "$PROXY_VAL_SPLIT" \
        --test-split "$OFFICIAL_VAL_SPLIT" --test-loss-csv "$VAL_LOSS_CSV" \
        --test-bpp-csv "$VAL_BPP" --init-checkpoint "$PROXY_INIT" \
        --out-dir "$RATE_PROXY_OUT" --lambdas $LAMBDAS \
        --target-scale 1.0 --voxel-size 0.16 0.16 0.16 \
        --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0 \
        --max-voxels 50000 --feat-dim 256 --epochs 60 \
        --batch-size 4 --workers 2 --lr 4e-4 --weight-decay 5e-4 \
        --rate-weight 1.0 --rd-weight 0.0 --selection-temperature 1.0 \
        --jitter-std 0.005 --patience 12 --seed 20260825 \
        >"$RATE_PROXY_OUT/router_train.log" 2>&1
fi
[[ -s "$RATE_PROXY_OUT/TRAINING_COMPLETE.json" ]]
[[ -s "$RATE_PROXY_OUT/test_rate_aware_predictions.csv" ]]
[[ $(find "$PREDICTION_ROOT" -type f -name predictions.manifest.json | wc -l) -eq "$PRED_SHARDS" ]]
touch "$PREDICTION_ROOT/PREDICTIONS_COMPLETE"

record "Stage 5/5: evaluate official validation mAP-BPP and render the final PNG"
PROXY_EPOCH=$("$PYTHON" -c "import torch; print(torch.load('$RATE_PROXY_OUT/best.pth',map_location='cpu')['epoch'])")
"$PYTHON" -u "$EVALUATOR" \
    --config "$CONFIG" --prediction-root "$PREDICTION_ROOT" \
    --rate-aware-predictions-csv "$RATE_PROXY_OUT/test_rate_aware_predictions.csv" \
    --output-dir "$MAP_OUT" --checkpoint-epoch "$PROXY_EPOCH" \
    --parallel-workers 12
[[ -s "$MAP_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.csv" ]]
[[ -s "$MAP_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.png" ]]

touch "$BASE/${TAG}_ALL_DONE"
record "ALL DONE: seven-GPU multi-sweep loss export, loss+BPP router training, validation test, and mAP-BPP plot"
