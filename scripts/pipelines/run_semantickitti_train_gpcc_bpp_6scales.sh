#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/DATA/sm/RACO-LPCC
PYTHON=/home/sm/miniconda3/envs/SparsePCGC/bin/python
SCRIPT="$ROOT/scripts/label_generation/measure_semantickitti_train_gpcc_bpp_6scales.py"
CFG="$ROOT/extension/kitti.cfg"
ROUTER_DATA="$ROOT/mmdetection3d/work_dirs/semantickitti_xyz19_quantized_loss_labels_20260817/router_data"
POINTS="$ROUTER_DATA/velodyne"
SPLIT="$ROUTER_DATA/train.txt"
OUT=/public/DATA/sm/semantickitti_train_gpcc_2048_1024_512_256_128_64_20260823
SHARD_ROOT="$OUT/shards"
MERGED="$OUT/semantickitti_train_gpcc_per_frame_per_rate.csv"
MANIFEST="$OUT/semantickitti_train_gpcc_per_frame_per_rate.manifest.json"
TMP_ROOT=/tmp/sm_semantickitti_train_gpcc_20260823
LOG="$OUT/pipeline.log"
STATUS="$OUT/status.txt"
NUM_SHARDS=16

mkdir -p "$OUT" "$SHARD_ROOT" "$TMP_ROOT"
exec 9>"$OUT/.pipeline.lock"
if ! flock -n 9; then
    echo "SemanticKITTI train G-PCC pipeline is already active."
    exit 0
fi
exec >>"$LOG" 2>&1
rm -f "$OUT/FAILED" "$OUT/ALL_DONE"

record() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}
fail() {
    local code=$?
    record "FAILED exit=$code command=$BASH_COMMAND"
    touch "$OUT/FAILED"
    exit "$code"
}
trap fail ERR

record "Preflight: resumable SemanticKITTI train BPP measurement"
[[ -s "$SCRIPT" && -s "$CFG" && -s "$SPLIT" ]]
[[ -d "$POINTS" ]]
[[ "$(wc -l <"$SPLIT")" -eq 19130 ]]
[[ "$(find -L "$POINTS" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 19130 ]]
available_kib=$(df --output=avail -k /public | tail -n 1 | tr -dc '0-9')
[[ -n "$available_kib" && "$available_kib" -gt 2097152 ]]

record "Encoding 19,130 frames x 6 rates with $NUM_SHARDS CPU workers"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_dir="$SHARD_ROOT/shard_$(printf '%02d' "$shard")"
    mkdir -p "$shard_dir" "$TMP_ROOT/shard_$(printf '%02d' "$shard")"
    (
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        nice -n 5 "$PYTHON" -u "$SCRIPT" \
            --points-dir "$POINTS" --split-file "$SPLIT" \
            --output "$shard_dir/gpcc.csv" \
            --tmp-dir "$TMP_ROOT/shard_$(printf '%02d' "$shard")" \
            --cfg "$CFG" --shard-id "$shard" --num-shards "$NUM_SHARDS" \
            --log-every 10 >"$shard_dir/run.log" 2>&1
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]]

record "Merging and validating all persisted shard rows"
"$PYTHON" -u "$SCRIPT" \
    --points-dir "$POINTS" --split-file "$SPLIT" \
    --output "$MERGED" --tmp-dir "$TMP_ROOT" --cfg "$CFG" \
    --merge-root "$SHARD_ROOT" --expected-frames 19130
[[ "$(wc -l <"$MERGED")" -eq 114781 ]]
[[ -s "$OUT/semantickitti_train_gpcc_average.csv" ]]
[[ -s "$MANIFEST" ]]

touch "$OUT/ALL_DONE"
record "ALL DONE: per-frame six-rate SemanticKITTI train BPP saved and validated"
