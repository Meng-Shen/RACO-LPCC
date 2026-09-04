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
CONFIG_ROOT="$SCRIPT_DIR/../configs"
LABELS="$BASE/labels"
CHECKPOINTS="$BASE/checkpoints"
EXPERIMENTS="$BASE/experiments"
SOURCE=sm@219.223.200.160
SSH_KEY=/home/sm/.ssh/id_ed25519_node160_sync
SSH_OPTS="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
CONFIG="$CONFIG_ROOT/centerpoint/centerpoint_voxel01_xyz_singleframe_recovery_12e_nus-3d.py"
OFFICIAL_CKPT="$CHECKPOINTS/centerpoint_voxel01_circlenms_nuscenes_official.pth"
OFFICIAL_URL=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
DETECTOR_OUT="$EXPERIMENTS/centerpoint_voxel01_xyz_singleframe_recovery_12e"
TRAIN_LOSS_ROOT="$LABELS/train_quant_losses"
VAL_LOSS_ROOT="$LABELS/val_quant_losses"
TRAIN_LOSS_CSV="$TRAIN_LOSS_ROOT/train_losses_merged.csv"
VAL_LOSS_CSV="$VAL_LOSS_ROOT/val_losses_merged.csv"
PROXY_TRAIN_SPLIT="$TRAIN_LOSS_ROOT/proxy_train_tokens.txt"
PROXY_VAL_SPLIT="$TRAIN_LOSS_ROOT/proxy_val_tokens.txt"
OFFICIAL_VAL_SPLIT="$VAL_LOSS_ROOT/official_val_tokens.txt"
LAMBDA_JSON="$TRAIN_LOSS_ROOT/rd_lambdas_train_only.json"
TRAIN_BPP="$LABELS/nuscenes_train_gpcc_per_frame_per_rate.csv"
VAL_BPP="$LABELS/nuscenes_val_gpcc_per_frame_per_rate.csv"
RATE_PROXY_OUT="$EXPERIMENTS/nuscenes_rate_aware_from_kitti_rate_aware"
QSTEPS=2048,1024,512,256,128,64
LOG="$BASE/recovery_pipeline.log"
STATUS="$BASE/recovery_status.txt"

mkdir -p "$CODE" "$CONFIG_ROOT/centerpoint" "$LABELS" "$CHECKPOINTS" "$EXPERIMENTS" \
    "$TRAIN_LOSS_ROOT" "$VAL_LOSS_ROOT" "$RATE_PROXY_OUT"
exec 9>"$BASE/.recovery_pipeline.lock"
if ! flock -n 9; then
    echo "Another nuScenes recovery pipeline is already active."
    exit 0
fi
exec >>"$LOG" 2>&1
rm -f "$BASE/RECOVERY_FAILED"

record() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}
fail() {
    local code=$?
    record "FAILED exit=$code command=$BASH_COMMAND"
    touch "$BASE/RECOVERY_FAILED"
    exit "$code"
}
trap fail ERR

gpu_memory_limit=700
free_gpu_csv() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | \
        awk -F',' -v limit="$gpu_memory_limit" \
            '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 < limit && $3 < 10) print $1}' | \
        head -n 4 | paste -sd, -
}
wait_for_gpu() {
    local list
    while true; do
        list=$(free_gpu_csv)
        if [[ -n "$list" ]]; then
            printf '%s\n' "$list"
            return
        fi
        record "Waiting for at least one idle GPU" >&2
        sleep 60
    done
}
gpu_count() {
    awk -F',' '{print NF}' <<<"$1"
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH="$CODE:$MIM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

record "Stage 1/7: waiting for persistent asset synchronization"
while [[ ! -f "$BASE/SYNC_COMPLETE" ]]; do
    if [[ -f "$BASE/SYNC_FAILED" ]]; then
        exit 1
    fi
    sleep 30
done
[[ -s "$DATA/v1.0-trainval/sample.json" ]]
[[ -s "$CHECKPOINTS/kitti_rate_aware_5loss_plus_bpp_best.pth" ]]

record "Stage 2/7: obtaining official CenterPoint checkpoint and train BPP labels"
if [[ ! -s "$OFFICIAL_CKPT" ]]; then
    curl --noproxy '*' -fL --retry 8 --retry-delay 10 \
        -o "$OFFICIAL_CKPT.partial" "$OFFICIAL_URL"
    mv "$OFFICIAL_CKPT.partial" "$OFFICIAL_CKPT"
fi
while ! $SSH_OPTS "$SOURCE" \
    'test -f /public/DATA/sm/nuscenes_gpcc_train_2048_1024_512_256_128_64/ALL_DONE'; do
    record "Waiting for 28130-frame train G-PCC labels"
    sleep 60
done
rsync -a --partial -e "$SSH_OPTS" \
    "$SOURCE:/public/DATA/sm/nuscenes_gpcc_train_2048_1024_512_256_128_64/nuscenes_train_gpcc_per_frame_per_rate.csv" \
    "$TRAIN_BPP"
[[ -s "$TRAIN_BPP" && -s "$VAL_BPP" ]]

record "Stage 3/7: preparing single-keyframe XYZ-only nuScenes infos and GT database"
cp -a "$MIM/configs/." "$CONFIG_ROOT/"
if [[ ! -f "$DATA/.xyz_singleframe_prepared" ]]; then
    ulimit -n 65535 || true
    "$PYTHON" -u "$CODE/prepare_nuscenes_xyz_singleframe.py" \
        --root-path "$DATA" --workers 12
fi
[[ -s "$DATA/nuscenes_infos_train.pkl" ]]
[[ -s "$DATA/nuscenes_infos_val.pkl" ]]
[[ -s "$DATA/nuscenes_xyz_dbinfos_train.pkl" ]]

record "Stage 4/7: recovering the XYZ-only CenterPoint detector"
if [[ ! -f "$DETECTOR_OUT/DETECTOR_COMPLETE" ]]; then
    gpu_list=$(wait_for_gpu)
    nproc=$(gpu_count "$gpu_list")
    record "CenterPoint training on idle GPUs=$gpu_list"
    export CUDA_VISIBLE_DEVICES="$gpu_list"
    if (( nproc > 1 )); then
        "$PYTHON" -m torch.distributed.launch --use_env \
            --nproc_per_node="$nproc" --master_port=29710 \
            "$MIM/tools/train.py" "$CONFIG" --launcher pytorch
    else
        "$PYTHON" "$MIM/tools/train.py" "$CONFIG"
    fi
    detector_best=$(find "$DETECTOR_OUT" -maxdepth 1 -type f -name 'best*.pth' \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "$detector_best" ]]; then
        detector_best="$DETECTOR_OUT/epoch_12.pth"
    fi
    [[ -s "$detector_best" ]]
    printf '%s\n' "$detector_best" >"$DETECTOR_OUT/best_checkpoint.txt"
    touch "$DETECTOR_OUT/DETECTOR_COMPLETE"
fi
DETECTOR_CKPT=$(tr -d '\r\n' <"$DETECTOR_OUT/best_checkpoint.txt")
[[ -s "$DETECTOR_CKPT" ]]

record "Stage 5/7: exporting six-rate train and official-val detector losses"
if [[ ! -f "$VAL_LOSS_ROOT/LOSSES_COMPLETE" ]]; then
    gpu_list=$(wait_for_gpu)
    export_gpus=(0 0 0 0 0 0 3 4 5 6)
    nshards=${#export_gpus[@]}
    record "Loss export GPU map=${export_gpus[*]} shards=$nshards"
    pids=()
    for ((shard=0; shard<nshards; shard++)); do
        gpu=${export_gpus[$shard]}
        train_dir="$TRAIN_LOSS_ROOT/shard_$shard"
        val_dir="$VAL_LOSS_ROOT/shard_$shard"
        mkdir -p "$train_dir" "$val_dir"
        (
            if [[ ! -s "$train_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
                    "$CODE/export_nuscenes_centerpoint_quant.py" \
                    --mode loss --config "$CONFIG" --checkpoint "$DETECTOR_CKPT" \
                    --data-root "$DATA" --split train --qsteps-mm "$QSTEPS" \
                    --output "$train_dir/loss.csv" --shard-id "$shard" \
                    --num-shards "$nshards" --device cuda:0 \
                    >"$train_dir/export.log" 2>&1
            fi
            if [[ ! -s "$val_dir/loss.manifest.json" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
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

record "Stage 6/7: merging loss labels and calibrating train-only lambdas"
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
LAMBDAS=$($PYTHON -c \
    "import json; print(' '.join(map(str,json.load(open('$LAMBDA_JSON'))['lambdas_high_rate_to_low_rate'])))")

record "Stage 7/7: fine-tuning transferred five loss heads plus BPP head"
if [[ ! -s "$RATE_PROXY_OUT/TRAINING_COMPLETE.json" ]]; then
    gpu_list=$(wait_for_gpu)
    gpu=${gpu_list%%,*}
    record "Rate-aware proxy fine-tuning on idle GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
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
        --jitter-std 0.005 --patience 12 --seed 20260822
fi

touch "$BASE/ALL_DONE"
record "ALL DONE: KITTI rate-aware initialization fine-tuned on nuScenes"
