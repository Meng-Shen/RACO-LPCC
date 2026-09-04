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
CHECKPOINTS="$BASE/checkpoints"
TAG=map_selected_from_epoch6_6e_20260823
CONFIG="$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_singleframe_${TAG}_nus-3d.py"
DETECTOR_OUT="$BASE/experiments/centerpoint_voxel01_xyz_singleframe_${TAG}"
DETECTOR_SELECTION="$DETECTOR_OUT/best_overall_map.json"
INITIAL_DETECTOR_OUT="$BASE/experiments/centerpoint_voxel01_xyz_singleframe_recovery_12e"
INITIAL_DETECTOR_CKPT="$INITIAL_DETECTOR_OUT/epoch_6.pth"
INITIAL_SELECTION="$DETECTOR_OUT/epoch6_initialization_overall_map.json"
RUN_LABELS="$BASE/labels/$TAG"
TRAIN_LOSS_ROOT="$RUN_LABELS/train_quant_losses"
VAL_LOSS_ROOT="$RUN_LABELS/val_quant_losses"
TRAIN_LOSS_CSV="$TRAIN_LOSS_ROOT/train_losses_merged.csv"
VAL_LOSS_CSV="$VAL_LOSS_ROOT/val_losses_merged.csv"
PROXY_TRAIN_SPLIT="$TRAIN_LOSS_ROOT/proxy_train_tokens.txt"
PROXY_VAL_SPLIT="$TRAIN_LOSS_ROOT/proxy_val_tokens.txt"
OFFICIAL_VAL_SPLIT="$VAL_LOSS_ROOT/official_val_tokens.txt"
LAMBDA_JSON="$TRAIN_LOSS_ROOT/rd_lambdas_train_only.json"
TRAIN_BPP="$BASE/labels/nuscenes_train_gpcc_per_frame_per_rate.csv"
VAL_BPP="$BASE/labels/nuscenes_val_gpcc_per_frame_per_rate.csv"
RATE_PROXY_OUT="$BASE/experiments/nuscenes_rate_aware_${TAG}"
PREDICTION_ROOT="$RATE_PROXY_OUT/val_prediction_cache"
MAP_OUT="$RATE_PROXY_OUT/final_map_bpp"
QSTEPS=2048,1024,512,256,128,64
LOG="$BASE/${TAG}_pipeline.log"
STATUS="$BASE/${TAG}_status.txt"

mkdir -p "$DETECTOR_OUT" "$TRAIN_LOSS_ROOT" "$VAL_LOSS_ROOT" \
    "$RATE_PROXY_OUT" "$PREDICTION_ROOT" "$MAP_OUT"
exec 9>"$BASE/.${TAG}.lock"
if ! flock -n 9; then
    echo "The corrected nuScenes mAP-selected pipeline is already active."
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

wait_for_gpu0() {
    local used util
    while true; do
        IFS=',' read -r used util < <(nvidia-smi -i 0 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' ')
        if (( used < 1000 && util < 15 )); then
            return
        fi
        record "Waiting for GPU 0: memory=${used}MiB utilization=${util}%"
        sleep 60
    done
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

record "Stage 1/7: preflight and metric-selection guard"
[[ -s "$CONFIG" ]]
[[ -s "$DATA/nuscenes_infos_train.pkl" && -s "$DATA/nuscenes_infos_val.pkl" ]]
[[ -s "$INITIAL_DETECTOR_CKPT" ]]
[[ -s "$CHECKPOINTS/kitti_rate_aware_5loss_plus_bpp_best.pth" ]]
[[ -s "$TRAIN_BPP" && -s "$VAL_BPP" ]]
"$PYTHON" -c "from mmengine.config import Config; c=Config.fromfile('$CONFIG'); assert c.train_cfg.max_epochs == 6; assert c.train_cfg.val_interval == 1; assert c.load_from == '$INITIAL_DETECTOR_CKPT'; assert c.resume is False; h=c.default_hooks.checkpoint; assert h.save_best == 'NuScenes metric/pred_instances_3d_NuScenes/mAP'; assert h.rule == 'greater'; assert c.work_dir == '$DETECTOR_OUT'; print('preflight: epoch-6 initialization, six additional epochs, overall mAP save_best')"

record "Stage 2/7: initialize from old epoch 6 and fine-tune six additional epochs on GPU 0"
if [[ ! -f "$DETECTOR_OUT/DETECTOR_COMPLETE" ]]; then
    "$PYTHON" "$CODE/select_best_nuscenes_checkpoint.py" \
        --work-dir "$INITIAL_DETECTOR_OUT" --output-json "$INITIAL_SELECTION"
    wait_for_gpu0
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$MIM/tools/train.py" \
        "$CONFIG" --work-dir "$DETECTOR_OUT"
    "$PYTHON" "$CODE/select_best_nuscenes_checkpoint.py" \
        --work-dir "$DETECTOR_OUT" --output-json "$DETECTOR_SELECTION" \
        --baseline-selection-json "$INITIAL_SELECTION"
    DETECTOR_CKPT=$("$PYTHON" -c "import json; print(json.load(open('$DETECTOR_SELECTION'))['checkpoint'])")
    DETECTOR_METRIC=$("$PYTHON" -c "import json; x=json.load(open('$DETECTOR_SELECTION')); print(f\"source={x['selected_source']} epoch={x['best_epoch']} overall_mAP={x['best_mAP']:.6f}\")")
    [[ -s "$DETECTOR_CKPT" ]]
    [[ "$(basename "$DETECTOR_CKPT")" == epoch_*.pth ]]
    [[ "$(basename "$DETECTOR_CKPT")" != *car_AP* ]]
    printf '%s\n' "$DETECTOR_CKPT" >"$DETECTOR_OUT/best_checkpoint.txt"
    printf '%s\n' 'NuScenes metric/pred_instances_3d_NuScenes/mAP' \
        >"$DETECTOR_OUT/selection_metric.txt"
    touch "$DETECTOR_OUT/DETECTOR_COMPLETE"
    record "Detector selected only by official overall mAP: $DETECTOR_METRIC"
fi
DETECTOR_CKPT=$(tr -d '\r\n' <"$DETECTOR_OUT/best_checkpoint.txt")
[[ -s "$DETECTOR_SELECTION" && -s "$DETECTOR_CKPT" ]]
[[ "$(tr -d '\r\n' <"$DETECTOR_OUT/selection_metric.txt")" == \
    'NuScenes metric/pred_instances_3d_NuScenes/mAP' ]]
[[ "$(basename "$DETECTOR_CKPT")" == epoch_*.pth ]]
[[ "$(basename "$DETECTOR_CKPT")" != *car_AP* ]]

record "Stage 3/7: recompute six-rate train/val detector losses with selected detector"
if [[ ! -f "$VAL_LOSS_ROOT/LOSSES_COMPLETE" ]]; then
    wait_for_gpu0
    nshards=4
    pids=()
    for ((shard=0; shard<nshards; shard++)); do
        train_dir="$TRAIN_LOSS_ROOT/shard_$shard"
        val_dir="$VAL_LOSS_ROOT/shard_$shard"
        mkdir -p "$train_dir" "$val_dir"
        (
            if [[ ! -s "$train_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
                    "$CODE/export_nuscenes_centerpoint_quant.py" \
                    --mode loss --config "$CONFIG" --checkpoint "$DETECTOR_CKPT" \
                    --data-root "$DATA" --split train --qsteps-mm "$QSTEPS" \
                    --output "$train_dir/loss.csv" --shard-id "$shard" \
                    --num-shards "$nshards" --device cuda:0 \
                    >"$train_dir/export.log" 2>&1
            fi
            if [[ ! -s "$val_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
                    "$CODE/export_nuscenes_centerpoint_quant.py" \
                    --mode loss --config "$CONFIG" --checkpoint "$DETECTOR_CKPT" \
                    --data-root "$DATA" --split val --qsteps-mm "$QSTEPS" \
                    --output "$val_dir/loss.csv" --shard-id "$shard" \
                    --num-shards "$nshards" --device cuda:0 \
                    >"$val_dir/export.log" 2>&1
            fi
        ) &
        pids+=("$!")
    done
    export_failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || export_failed=1
    done
    [[ "$export_failed" == 0 ]]
    touch "$VAL_LOSS_ROOT/LOSSES_COMPLETE"
fi

record "Stage 4/7: merge labels and calibrate lambdas on proxy training split only"
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

record "Stage 5/7: train rate-aware proxy from KITTI loss+BPP initialization"
if [[ ! -s "$RATE_PROXY_OUT/TRAINING_COMPLETE.json" ]]; then
    wait_for_gpu0
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
        "$CODE/train_scannet_rate_aware_proxy.py" \
        --dataset-format nuscenes --points-dir "$DATA" \
        --train-loss-csv "$TRAIN_LOSS_CSV" --val-loss-csv "$TRAIN_LOSS_CSV" \
        --train-bpp-csv "$TRAIN_BPP" --val-bpp-csv "$TRAIN_BPP" \
        --train-split "$PROXY_TRAIN_SPLIT" --val-split "$PROXY_VAL_SPLIT" \
        --test-split "$OFFICIAL_VAL_SPLIT" --test-loss-csv "$VAL_LOSS_CSV" \
        --test-bpp-csv "$VAL_BPP" \
        --init-checkpoint "$CHECKPOINTS/kitti_rate_aware_5loss_plus_bpp_best.pth" \
        --out-dir "$RATE_PROXY_OUT" --lambdas $LAMBDAS \
        --target-scale 1.0 --voxel-size 0.16 0.16 0.16 \
        --point-cloud-range -51.2 -51.2 -5.0 51.2 51.2 3.0 \
        --max-voxels 50000 --feat-dim 256 --epochs 60 \
        --batch-size 8 --workers 4 --lr 2e-4 --weight-decay 5e-4 \
        --rate-weight 1.0 --rd-weight 0.0 --selection-temperature 1.0 \
        --jitter-std 0.005 --patience 12 --seed 20260823
fi
[[ -s "$RATE_PROXY_OUT/test_rate_aware_predictions.csv" ]]

record "Stage 6/7: cache six-rate validation detections from the selected detector"
if [[ ! -f "$PREDICTION_ROOT/PREDICTIONS_COMPLETE" ]]; then
    wait_for_gpu0
    nshards=4
    pids=()
    for ((shard=0; shard<nshards; shard++)); do
        shard_dir="$PREDICTION_ROOT/shard_$shard"
        mkdir -p "$shard_dir"
        (
            if [[ ! -s "$shard_dir/predictions.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
                    "$CODE/export_nuscenes_centerpoint_quant.py" \
                    --mode predictions --config "$CONFIG" \
                    --checkpoint "$DETECTOR_CKPT" --data-root "$DATA" \
                    --split val --qsteps-mm "$QSTEPS" \
                    --output "$shard_dir/predictions.pkl" \
                    --shard-id "$shard" --num-shards "$nshards" \
                    --device cuda:0 >"$shard_dir/export.log" 2>&1
            fi
        ) &
        pids+=("$!")
    done
    export_failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || export_failed=1
    done
    [[ "$export_failed" == 0 ]]
    [[ "$(find "$PREDICTION_ROOT" -type f -name predictions.pkl | wc -l)" -eq 4 ]]
    touch "$PREDICTION_ROOT/PREDICTIONS_COMPLETE"
fi

record "Stage 7/7: evaluate official nuScenes mAP-BPP and render PNG"
PROXY_EPOCH=$("$PYTHON" -c "import torch; print(torch.load('$RATE_PROXY_OUT/best.pth', map_location='cpu')['epoch'])")
"$PYTHON" -u "$CODE/evaluate_nuscenes_rate_aware_map_bpp.py" \
    --config "$CONFIG" --prediction-root "$PREDICTION_ROOT" \
    --rate-aware-predictions-csv "$RATE_PROXY_OUT/test_rate_aware_predictions.csv" \
    --output-dir "$MAP_OUT" --checkpoint-epoch "$PROXY_EPOCH" \
    --parallel-workers 12
[[ -s "$MAP_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.csv" ]]
[[ -s "$MAP_OUT/nuscenes_rate_aware_measured_gpcc_map_bpp.png" ]]

touch "$BASE/${TAG}_ALL_DONE"
record "ALL DONE: epoch-6 initialized detector selected by official overall mAP; labels, proxy, and mAP-BPP rebuilt"
