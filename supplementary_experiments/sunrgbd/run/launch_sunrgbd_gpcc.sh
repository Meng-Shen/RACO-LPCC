#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
DATA="$ROOT/data/sunrgbd"
OUT="$ROOT/labels/gpcc"
SHARDS=14

mkdir -p "$OUT" "$ROOT/logs/gpcc" "$ROOT/tmp/gpcc"
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  (
    mkdir -p "$OUT/train/shard_$shard" "$OUT/val/shard_$shard" "$ROOT/tmp/gpcc/shard_$shard"
    "$PY" "$CODE/measure_sunrgbd_gpcc.py" \
      --points-dir "$DATA/points" \
      --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
      --split-name train \
      --output "$OUT/train/shard_$shard/gpcc.csv" \
      --tmp-dir "$ROOT/tmp/gpcc/shard_$shard" \
      --tmc3 "$ROOT/bin/tmc3_v22" --config "$ROOT/bin/dense.cfg" \
      --shard-id "$shard" --num-shards "$SHARDS"
    "$PY" "$CODE/measure_sunrgbd_gpcc.py" \
      --points-dir "$DATA/points" \
      --split-file "$DATA/sunrgbd_trainval/val_data_idx.txt" \
      --split-name val \
      --output "$OUT/val/shard_$shard/gpcc.csv" \
      --tmp-dir "$ROOT/tmp/gpcc/shard_$shard" \
      --tmc3 "$ROOT/bin/tmc3_v22" --config "$ROOT/bin/dense.cfg" \
      --shard-id "$shard" --num-shards "$SHARDS"
  ) > "$ROOT/logs/gpcc/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" "$CODE/measure_sunrgbd_gpcc.py" \
  --merge-root "$OUT/train" \
  --split-file "$DATA/sunrgbd_trainval/train_data_idx.txt" \
  --expected-scenes 5285 --output "$OUT/sunrgbd_train_gpcc.csv"
"$PY" "$CODE/measure_sunrgbd_gpcc.py" \
  --merge-root "$OUT/val" \
  --split-file "$DATA/sunrgbd_trainval/val_data_idx.txt" \
  --expected-scenes 5050 --output "$OUT/sunrgbd_val_gpcc.csv"

date -Is > "$ROOT/state/GPCC_COMPLETE"
