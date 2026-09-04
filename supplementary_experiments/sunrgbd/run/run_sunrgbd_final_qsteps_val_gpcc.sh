#!/usr/bin/env bash
set -euo pipefail

ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
DATA="$ROOT/data/sunrgbd"
STATE="$ROOT/state"
EXTRA="$ROOT/supplemental/val_qsteps_120_100_60"
FINAL="$ROOT/final_qsteps_160_120_100_80_60_40"
LOGS="$ROOT/logs/final_val_gpcc"
STAGE=encoding

mkdir -p "$STATE" "$EXTRA/gpcc" "$EXTRA/tmp" "$FINAL" "$LOGS"
trap 'printf "{\"status\":\"failed\",\"stage\":\"%s\",\"time\":\"%s\"}\n" "$STAGE" "$(date -Is)" > "$STATE/FINAL_QSTEPS_VAL_GPCC_FAILED.json"' ERR
date -Is > "$STATE/FINAL_QSTEPS_VAL_GPCC_STARTED"

pids=()
for shard in $(seq 0 13); do
  (
    mkdir -p "$EXTRA/gpcc/shard_$shard" "$EXTRA/tmp/shard_$shard"
    "$PY" "$CODE/measure_sunrgbd_gpcc.py" \
      --points-dir "$DATA/points" \
      --split-file "$DATA/sunrgbd_trainval/val_data_idx.txt" \
      --split-name val --qsteps-mm 120 100 60 \
      --output "$EXTRA/gpcc/shard_$shard/gpcc.csv" \
      --tmp-dir "$EXTRA/tmp/shard_$shard" \
      --tmc3 "$ROOT/bin/tmc3_v22" --config "$ROOT/bin/dense.cfg" \
      --shard-id "$shard" --num-shards 14
  ) > "$LOGS/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

STAGE=merging_supplemental
"$PY" "$CODE/measure_sunrgbd_gpcc.py" \
  --merge-root "$EXTRA/gpcc" \
  --split-file "$DATA/sunrgbd_trainval/val_data_idx.txt" \
  --expected-scenes 5050 --qsteps-mm 120 100 60 \
  --output "$EXTRA/sunrgbd_val_gpcc_qsteps_120_100_60.csv" \
  > "$LOGS/merge_extra.log" 2>&1

STAGE=combining_final_six
"$PY" "$CODE/combine_sunrgbd_gpcc_qsteps.py" \
  --base "$ROOT/labels/gpcc/sunrgbd_val_gpcc.csv" \
  --extra "$EXTRA/sunrgbd_val_gpcc_qsteps_120_100_60.csv" \
  --output "$FINAL/sunrgbd_val_gpcc.csv" \
  --target-qsteps 160 120 100 80 60 40 --expected-scenes 5050 \
  > "$LOGS/combine_final.log" 2>&1

STAGE=complete
date -Is > "$STATE/FINAL_QSTEPS_VAL_GPCC_COMPLETE"
rm -f "$STATE/FINAL_QSTEPS_VAL_GPCC_FAILED.json"
