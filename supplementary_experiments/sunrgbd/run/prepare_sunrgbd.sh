#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/../code" && pwd)"

EXP_ROOT="${RACO_SUNRGBD_ROOT:-/home/sm/sunrgbd_lite_s3_20260828}"
RAW_DIR="$EXP_ROOT/data/raw/OFFICIAL_SUNRGBD"
DATA_ROOT="$EXP_ROOT/data/sunrgbd"
MMDET_ROOT="$EXP_ROOT/mmdetection3d"
LOG_DIR="$EXP_ROOT/logs"
STATE_DIR="$EXP_ROOT/state"

if [[ ! -f "$STATE_DIR/DOWNLOAD_COMPLETE" ]]; then
  echo "DOWNLOAD_COMPLETE is missing" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT/matlab" "$LOG_DIR" "$STATE_DIR"

if [[ ! -f "$STATE_DIR/RAW_EXTRACT_COMPLETE" ]]; then
  unzip -t "$RAW_DIR/SUNRGBD.zip" >/dev/null
  unzip -t "$RAW_DIR/SUNRGBDtoolbox.zip" >/dev/null
  unzip -q -o "$RAW_DIR/SUNRGBD.zip" -d "$RAW_DIR"
  unzip -q -o "$RAW_DIR/SUNRGBDtoolbox.zip" -d "$RAW_DIR"
  test -d "$RAW_DIR/SUNRGBD"
  test -d "$RAW_DIR/SUNRGBDtoolbox"
  date -Is > "$STATE_DIR/RAW_EXTRACT_COMPLETE"
fi

if [[ ! -e "$DATA_ROOT/OFFICIAL_SUNRGBD" ]]; then
  ln -s "$RAW_DIR" "$DATA_ROOT/OFFICIAL_SUNRGBD"
fi

if [[ ! -f "$STATE_DIR/SPLIT_EXTRACT_COMPLETE" ]]; then
  /home/sm/miniconda3/envs/openmmlab/bin/python \
    "$CODE/extract_sunrgbd_split.py" \
    --toolbox "$RAW_DIR/SUNRGBDtoolbox" \
    --meta3d "$RAW_DIR/SUNRGBDMeta3DBB_v2.mat" \
    --output-dir "$DATA_ROOT/sunrgbd_trainval" \
    > "$LOG_DIR/extract_split.log" 2>&1
  test "$(wc -l < "$DATA_ROOT/sunrgbd_trainval/train_data_idx.txt")" -eq 5285
  test "$(wc -l < "$DATA_ROOT/sunrgbd_trainval/val_data_idx.txt")" -eq 5050
  date -Is > "$STATE_DIR/SPLIT_EXTRACT_COMPLETE"
fi

if [[ ! -f "$STATE_DIR/RGBD_EXTRACT_COMPLETE" ]]; then
  pids=()
  for shard in $(seq 0 6); do
    /home/sm/miniconda3/envs/openmmlab/bin/python \
      "$CODE/extract_sunrgbd_rgbd_python.py" \
      --raw-root "$RAW_DIR" \
      --meta3d "$RAW_DIR/SUNRGBDMeta3DBB_v2.mat" \
      --meta2d "$RAW_DIR/SUNRGBDMeta2DBB_v2.mat" \
      --output-root "$DATA_ROOT/sunrgbd_trainval" \
      --shard-id "$shard" --num-shards 7 \
      > "$LOG_DIR/extract_rgbd_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  test "$(find "$DATA_ROOT/sunrgbd_trainval/depth" -maxdepth 1 -type f -name '*.mat' | wc -l)" -eq 10335
  test "$(find "$DATA_ROOT/sunrgbd_trainval/calib" -maxdepth 1 -type f -name '*.txt' | wc -l)" -eq 10335
  test "$(find "$DATA_ROOT/sunrgbd_trainval/label" -maxdepth 1 -type f -name '*.txt' | wc -l)" -eq 10335
  date -Is > "$STATE_DIR/RGBD_EXTRACT_COMPLETE"
fi

if [[ ! -f "$STATE_DIR/MMDET_DATA_COMPLETE" ]]; then
  cd "$MMDET_ROOT"
  PYTHONPATH="$MMDET_ROOT" /home/sm/miniconda3/envs/openmmlab/bin/python \
    tools/create_data.py sunrgbd \
    --root-path "$DATA_ROOT" \
    --out-dir "$DATA_ROOT" \
    --extra-tag sunrgbd \
    --workers 16
  test -f "$DATA_ROOT/sunrgbd_infos_train.pkl"
  test -f "$DATA_ROOT/sunrgbd_infos_val.pkl"
  test "$(find "$DATA_ROOT/points" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 10335
  date -Is > "$STATE_DIR/MMDET_DATA_COMPLETE"
fi

date -Is > "$STATE_DIR/PREPARE_COMPLETE"
