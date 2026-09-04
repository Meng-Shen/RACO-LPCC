#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/DATA/sm/RACO-LPCC
PYTHON=/home/sm/miniconda3/envs/SparsePCGC/bin/python
SPLIT=${ROOT}/OpenPCDet/data/kitti_fov/ImageSets/train.txt
POINTS=${ROOT}/OpenPCDet/data/kitti_fov/training/velodyne
OUT=${ROOT}/experiment_results/kitti_detection_gpcc_training_bpp_6scales
CFG=${ROOT}/extension/kitti.cfg
SCALES=(1/2048 1/1024 1/512 1/256 1/128 1/64)

mkdir -p "${OUT}/shards"
printf '[%s] KITTI train G-PCC six-scale measurement started\n' "$(date '+%F %T')"

pids=()
for rate_id in "${!SCALES[@]}"; do
  scale=${SCALES[$rate_id]}
  shard=${OUT}/shards/scale_${rate_id}
  mkdir -p "${shard}/tmp"
  (
    cd "$ROOT"
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON" GPCC/baseline_rates.py \
        --testdata "$POINTS" \
        --split_file "$SPLIT" \
        --scales "$scale" \
        --results "$shard" \
        --tmp_dir "${shard}/tmp" \
        --cfg "$CFG"
  ) > "${shard}/run.log" 2>&1 &
  pids+=("$!")
  printf 'rate=%s scale=%s pid=%s\n' "$rate_id" "$scale" "$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
  printf '[%s] one or more G-PCC workers failed\n' "$(date '+%F %T')" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" GPCC/merge_uniform_gpcc_shards.py \
  --shard_root "${OUT}/shards" \
  --scales "1/2048,1/1024,1/512,1/256,1/128,1/64" \
  --split_file "$SPLIT" \
  --details_out "${OUT}/gpcc_train_details.csv" \
  --average_out "${OUT}/gpcc_train_average.csv"

printf '{"status":"complete","completed_at":"%s","frames":3712,"levels":6}\n' \
  "$(date --iso-8601=seconds)" > "${OUT}/GPCC_TRAIN_COMPLETE.json"
printf '[%s] KITTI train G-PCC six-scale measurement complete\n' "$(date '+%F %T')"
