#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm/raco_rate_aware_nuscenes_20260822
ENV=/home/sm/miniconda3/envs/openmmlab
PYTHON=$ENV/bin/python
MIM=$ENV/lib/python3.8/site-packages/mmdet3d/.mim
CODE=$ROOT/code/nuscenes_detector_transfer_20260825
CHECKPOINTS=$ROOT/checkpoints/transfer_detectors_20260825
OUT=$ROOT/experiments/nuscenes_centerpoint_route_transfer_detectors_20260825
DATA=$ROOT/data/nuscenes
EXPORTER=$ROOT/code/export_nuscenes_centerpoint_quant_multisweep.py
ROUTE_CSV=$ROOT/experiments/nuscenes_rate_aware_multisweep_cbgs_bestmap_epoch4_normloss_mapselect_7gpu_20260825/test_rate_aware_predictions_map_selected.csv
QSTEPS=2048,1024,512,256,128,64
LOG=$OUT/pipeline.log

mkdir -p "$OUT"
exec 9>"$OUT/pipeline.lock"
if ! flock -n 9; then
  echo "Another detector-transfer pipeline already owns the lock."
  exit 2
fi
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

record() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
require_file() {
  if [[ ! -s "$1" ]]; then
    record "ERROR: missing file $1"
    exit 1
  fi
}

require_file "$ROUTE_CSV"
require_file "$CHECKPOINTS/pointpillars_official.pth"
require_file "$CHECKPOINTS/ssn_official.pth"

prepare_checkpoint() {
  local name=$1
  local source=$2
  local config=$3
  local adapted=$4
  if [[ ! -s "$adapted" ]]; then
    record "Adapting $name official checkpoint to strict XYZ input"
    "$PYTHON" "$CODE/adapt_official_checkpoint_to_xyz.py" \
      --config "$config" --source "$source" --output "$adapted"
  fi
  require_file "$adapted"
}

train_detector() {
  local name=$1
  local config=$2
  local adapted=$3
  local port=$4
  local detector_out=$OUT/$name
  local train_out=$detector_out/geometry_finetune
  mkdir -p "$train_out"

  if [[ ! -s "$detector_out/SMOKE_PASSED.json" ]]; then
    record "$name: executing one real XYZ-only nuScenes loss step"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE/smoke_xyz_detector.py" \
      --config "$config" --checkpoint "$adapted" \
      --output "$detector_out/SMOKE_PASSED.json"
  fi

  if [[ ! -s "$detector_out/DETECTOR_TRAINING_COMPLETE.json" ]]; then
    record "$name: seven-GPU geometry-only fine-tuning"
    resume_args=()
    if [[ -s "$train_out/last_checkpoint" ]]; then
      resume_args=(--resume)
    fi
    "$ENV/bin/torchrun" --nnodes=1 --node_rank=0 --nproc_per_node=7 \
      --master_addr=127.0.0.1 --master_port="$port" \
      "$MIM/tools/train.py" "$config" --launcher pytorch \
      --work-dir "$train_out" "${resume_args[@]}"

    best=$(find "$train_out" -maxdepth 1 -type f -name 'best_*mAP*.pth' -print -quit)
    if [[ -z "$best" ]]; then
      record "ERROR: $name training produced no validation-mAP checkpoint"
      exit 1
    fi
    cp -f "$best" "$detector_out/best.pth"
    "$PYTHON" -c \
      "import json; from pathlib import Path; Path('$detector_out/DETECTOR_TRAINING_COMPLETE.json').write_text(json.dumps({'status':'complete','detector':'$name','best_source':'$best','best_copy':'$detector_out/best.pth','selection_metric':'official nuScenes validation mAP','geometry_only':True,'test_used_for_selection':False,'gpus':7},indent=2))"
  fi
  require_file "$detector_out/best.pth"
}

cache_predictions() {
  local name=$1
  local config=$2
  local detector_out=$OUT/$name
  local pred_root=$detector_out/val_prediction_cache
  if [[ -s "$detector_out/PREDICTIONS_COMPLETE.json" ]]; then
    return
  fi
  record "$name: caching six quantization-level predictions on seven GPUs"
  mkdir -p "$pred_root"
  pids=()
  for shard in 0 1 2 3 4 5 6; do
    mkdir -p "$pred_root/shard_$shard"
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON" -u "$EXPORTER" \
      --mode predictions --config "$config" \
      --checkpoint "$detector_out/best.pth" \
      --data-root "$DATA" --split val --qsteps-mm "$QSTEPS" \
      --output "$pred_root/shard_$shard/predictions.pkl" \
      --shard-id "$shard" --num-shards 7 --device cuda:0 \
      --log-every 50 \
      > "$pred_root/shard_$shard/export.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    record "ERROR: one or more $name prediction shards failed"
    exit 1
  fi
  for shard in 0 1 2 3 4 5 6; do
    require_file "$pred_root/shard_$shard/predictions.pkl"
  done
  "$PYTHON" -c \
    "import json; from pathlib import Path; Path('$detector_out/PREDICTIONS_COMPLETE.json').write_text(json.dumps({'status':'complete','shards':7,'quant_steps_mm':[2048,1024,512,256,128,64]},indent=2))"
}

evaluate_transfer() {
  local name=$1
  local display=$2
  local config=$3
  local detector_out=$OUT/$name
  local result=$detector_out/centerpoint_route_transfer
  if [[ ! -s "$result/summary.json" ]]; then
    record "$name: evaluating fixed baselines and transferred CenterPoint decisions"
    "$PYTHON" -u "$CODE/evaluate_nuscenes_transfer_map_bpp.py" \
      --config "$config" \
      --prediction-root "$detector_out/val_prediction_cache" \
      --rate-aware-predictions-csv "$ROUTE_CSV" \
      --output-dir "$result" --checkpoint-epoch 11 \
      --parallel-workers 6 --detector-name "$display" \
      --routing-name "CenterPoint routing transfer"
  fi
  require_file "$result/summary.json"
}

POINT_CFG=$CODE/pointpillars_xyz9sweeps_finetune.py
SSN_CFG=$CODE/ssn_xyz9sweeps_finetune.py
prepare_checkpoint PointPillars "$CHECKPOINTS/pointpillars_official.pth" \
  "$POINT_CFG" "$CHECKPOINTS/pointpillars_xyz_adapted.pth"
prepare_checkpoint SSN "$CHECKPOINTS/ssn_official.pth" \
  "$SSN_CFG" "$CHECKPOINTS/ssn_xyz_adapted.pth"

train_detector pointpillars "$POINT_CFG" "$CHECKPOINTS/pointpillars_xyz_adapted.pth" 29611
cache_predictions pointpillars "$POINT_CFG"
evaluate_transfer pointpillars PointPillars "$POINT_CFG"

train_detector ssn "$SSN_CFG" "$CHECKPOINTS/ssn_xyz_adapted.pth" 29612
cache_predictions ssn "$SSN_CFG"
evaluate_transfer ssn SSN "$SSN_CFG"

"$PYTHON" -c \
  "import json; from pathlib import Path; Path('$OUT/PIPELINE_COMPLETE.json').write_text(json.dumps({'status':'complete','detectors':['PointPillars','SSN'],'routing_source':'CenterPoint epoch-11 rate-aware decisions','geometry_only':True,'gpus':7},indent=2))"
record "nuScenes CenterPoint route transfer complete for PointPillars and SSN"
