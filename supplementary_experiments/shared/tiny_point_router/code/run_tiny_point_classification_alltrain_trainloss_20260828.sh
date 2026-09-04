#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/sm/tiny_point_router_20260828
PY=/home/sm/miniconda3/envs/openmmlab/bin/python
TRAINER="$ROOT/train_tiny_point_classification_alltrain.py"
MODELNET=/home/sm/raco_rate_aware_modelnet40_dgcnn_20260824
SHAPENET=/home/sm/raco_rate_aware_shapenet55_pointmae_20260825
MODELNET_OLD="$ROOT/modelnet40_tinypoint_sixloss_monotonic_20260828/best.pth"
SHAPENET_OLD="$ROOT/shapenet55_tinypoint_sixloss_monotonic_20260828/best.pth"
MODELNET_SMOKE="$ROOT/modelnet40_tinypoint_alltrain_trainloss_initbest_smoke_20260828"
SHAPENET_SMOKE="$ROOT/shapenet55_tinypoint_alltrain_trainloss_initbest_smoke_20260828"
MODELNET_OUT="$ROOT/modelnet40_tinypoint_alltrain_trainloss_initbest_20260828"
SHAPENET_OUT="$ROOT/shapenet55_tinypoint_alltrain_trainloss_initbest_20260828"
STATE="$ROOT/alltrain_trainloss_state_20260828"
LOG="$ROOT/alltrain_trainloss_pipeline_20260828.log"

mkdir -p "$MODELNET_SMOKE" "$SHAPENET_SMOKE" "$MODELNET_OUT" "$SHAPENET_OUT" "$STATE"
exec >>"$LOG" 2>&1

record() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}
fail() {
  code=$?
  printf '{"status":"failed","exit_code":%d,"command":"%s"}\n' "$code" "$BASH_COMMAND" >"$STATE/FAILED.json"
  record "FAILED exit=$code command=$BASH_COMMAND"
  exit "$code"
}
trap fail ERR

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export OMP_NUM_THREADS=1

modelnet_args=(
  --train-points "$MODELNET/data/train_points.npy"
  --test-points "$MODELNET/data/test_points.npy"
  --train-quant "$MODELNET/artifacts/train_quant_labels.npz"
  --test-quant "$MODELNET/artifacts/test_quant_labels.npz"
  --train-bpp "$MODELNET/artifacts/train_bpp.csv"
  --test-bpp "$MODELNET/artifacts/test_bpp.csv"
  --train-indices "$MODELNET/data/route_train_indices.npy"
  --val-indices "$MODELNET/data/route_val_indices.npy"
  --lambda-json "$MODELNET/artifacts/rd_lambdas_train_only.json"
  --init-checkpoint "$MODELNET_OLD" --init-kind tiny_point_full
  --dataset-name ModelNet40 --task-model DGCNN
  --selection-mode train_loss --merge-val-into-train
  --epochs 80 --patience 15 --batch-size 256 --workers 8
  --backbone-lr 5e-4 --head-lr 2.5e-4 --weight-decay 5e-4
  --loss-weight 2.0 --rate-weight 1.0 --seed 20260828
)

shapenet_args=(
  --train-points "$SHAPENET/data/all_points.npy"
  --test-points "$SHAPENET/data/all_points.npy"
  --train-quant "$SHAPENET/artifacts/all_quant_labels.npz"
  --test-quant "$SHAPENET/artifacts/all_quant_labels.npz"
  --train-bpp "$SHAPENET/artifacts/all_bpp.csv"
  --test-bpp "$SHAPENET/artifacts/all_bpp.csv"
  --train-indices "$SHAPENET/data/router_train_indices.npy"
  --val-indices "$SHAPENET/data/router_val_indices.npy"
  --test-indices "$SHAPENET/data/test_indices.npy"
  --lambda-json "$SHAPENET/artifacts/rd_lambdas_train_only.json"
  --init-checkpoint "$SHAPENET_OLD" --init-kind tiny_point_full
  --dataset-name ShapeNet55 --task-model Point-MAE
  --selection-mode train_loss --merge-val-into-train
  --epochs 80 --patience 15 --batch-size 256 --workers 8
  --backbone-lr 5e-4 --head-lr 2.5e-4 --weight-decay 5e-4
  --loss-weight 2.0 --rate-weight 1.0 --seed 20260828
)

record "Preflight"
"$PY" -m py_compile "$TRAINER"
for path in "$MODELNET_OLD" "$SHAPENET_OLD" \
  "$MODELNET/data/route_train_indices.npy" "$MODELNET/data/route_val_indices.npy" \
  "$SHAPENET/data/router_train_indices.npy" "$SHAPENET/data/router_val_indices.npy"; do
  [[ -s "$path" ]]
done

record "Parallel smoke tests"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$TRAINER" "${modelnet_args[@]}" \
  --output-dir "$MODELNET_SMOKE" --smoke-only >"$MODELNET_SMOKE/smoke.log" 2>&1 &
smoke_modelnet_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$TRAINER" "${shapenet_args[@]}" \
  --output-dir "$SHAPENET_SMOKE" --smoke-only >"$SHAPENET_SMOKE/smoke.log" 2>&1 &
smoke_shapenet_pid=$!
wait "$smoke_modelnet_pid"
wait "$smoke_shapenet_pid"
grep -q '"bpp_monotonic_violation_rate": 0.0' "$MODELNET_SMOKE/SMOKE_TEST.json"
grep -q '"bpp_monotonic_violation_rate": 0.0' "$SHAPENET_SMOKE/SMOKE_TEST.json"
touch "$STATE/SMOKE_COMPLETE"

record "Parallel full-training runs"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$TRAINER" "${modelnet_args[@]}" \
  --output-dir "$MODELNET_OUT" >"$MODELNET_OUT/train.log" 2>&1 &
modelnet_pid=$!
echo "$modelnet_pid" >"$STATE/MODELNET_PID"
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$TRAINER" "${shapenet_args[@]}" \
  --output-dir "$SHAPENET_OUT" >"$SHAPENET_OUT/train.log" 2>&1 &
shapenet_pid=$!
echo "$shapenet_pid" >"$STATE/SHAPENET_PID"
wait "$modelnet_pid"
touch "$STATE/MODELNET_COMPLETE"
wait "$shapenet_pid"
touch "$STATE/SHAPENET_COMPLETE"

[[ -s "$MODELNET_OUT/TRAINING_COMPLETE.json" ]]
[[ -s "$MODELNET_OUT/test_router_predictions.npz" ]]
[[ -s "$SHAPENET_OUT/TRAINING_COMPLETE.json" ]]
[[ -s "$SHAPENET_OUT/test_router_predictions.npz" ]]
printf '{"status":"complete","checkpoint_selection":"minimum full-training regression total loss","modelnet_all_training_samples":9840,"shapenet_all_training_samples":41952,"test_used_for_selection":false}\n' >"$STATE/PIPELINE_COMPLETE.json"
touch "$STATE/COMPLETE"
record "ModelNet40 and ShapeNet55 all-training TinyPoint retraining complete"
