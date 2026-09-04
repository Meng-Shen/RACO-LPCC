# RACO-LPCC

RACO-LPCC studies machine-task-aware point-cloud geometry compression. The
current checkout on `node-233` contains the maintained KITTI detection,
SemanticKITTI segmentation, LRProxy routing, G-PCC, and task-aware geometry
restoration code and artifacts.

This README describes the cleaned project state as of 2026-09-03. Removed
JUQP, Split-GPCC, Lite-S3, VF7, ScanNet, old calibration, and old
super-resolution experiments are intentionally not documented here.

## Current Experiment Contract

All current routing experiments use six whole-cloud geometry quantization
levels in fixed coarse-to-fine order:

| Rate ID | Quantization step | Fidelity |
| ---: | ---: | --- |
| 0 | 2048 mm | coarsest |
| 1 | 1024 mm | |
| 2 | 512 mm | |
| 3 | 256 mm | |
| 4 | 128 mm | |
| 5 | 64 mm | finest |

The rate for one operating point is always aggregated over the complete
dataset:

```text
BPP = sum(G-PCC coded bits over all frames) / sum(original points over all frames)
```

Per-frame BPP values are retained for routing, but averaging per-frame BPP is
not used as the reported dataset rate.

### LRproxy Router

The maintained router uses geometry only:

- Input features are normalized voxel-mean absolute XYZ (`LRproxy`).
- GPU voxelization uses 160 mm voxels by default.
- A shared LRProxy backbone feeds six independent absolute task-loss heads.
- All six task losses are predicted directly, including the finest 64 mm
  level. The loss outputs are not forced to be monotonic.
- The BPP head predicts a positive first rate and five positive adjacent
  increments. `softplus`, cumulative summation, and `expm1` guarantee that
  reconstructed BPP is nondecreasing from coarse to fine.
- There is no learned decision head.

The final rate is selected analytically:

```text
q*(x; lambda) = argmin_q [predicted_loss_q(x) + lambda * predicted_BPP_q(x)]
```

The current model has 540,460 trainable parameters: 76,832 in the LRProxy
backbone, 396,294 in the six independent loss heads, and 67,334 in the
monotonic BPP head.

KITTI and SemanticKITTI training use all designated training frames. The best
checkpoint is selected by minimum full-training regression loss; no holdout or
official test/validation task metric is used for checkpoint or lambda
selection.

### Task-Aware Coordinate Restorer

The maintained geometry-restoration module consumes G-PCC-decoded current
scale coordinates and one constant feature per active sparse voxel. It does
not use foreground/background labels, parent-scale tensors, occupancy-code
features, or extra transmitted side information.

Its main structure is:

```text
decoded q-level sparse coordinates + all-one features
    -> sparse feature extraction
    -> scale embedding / FiLM / low-rank scale expert
    -> sparse XYZ residual heads
    -> bounded coordinate reconstruction
```

The model has 113,736 trainable parameters and seven sparse convolution layers.
The final KITTI experiment applies restoration at 2048, 1024, 512, 256, and
128 mm. The 64 mm level is plain G-PCC passthrough, so its BPP and geometry are
not recomputed by the restorer. The restorer adds no coded bits; existing
G-PCC BPP is reused.

Detailed architecture notes are in:

```text
reno/current_q_ones_coordinate_v1_20260831/ARCHITECTURE.md
```

## Runtime

```text
Project root: /public/DATA/sm/RACO-LPCC
Python:       /home/sm/miniconda3/envs/SparsePCGC/bin/python
torchrun:     /home/sm/miniconda3/envs/SparsePCGC/bin/torchrun
gpustat:      /home/sm/miniconda3/bin/gpustat
Scratch:      /tmp/sm_storage
G-PCC binary: extension/tmc3_v22
G-PCC config: extension/kitti.cfg
```

Do not assume that a proxy forwarded through `127.0.0.1:7897` is available.

## Repository Layout

```text
RACO-LPCC/
├── OpenPCDet/       third-party KITTI detector source, data, and weights
├── mmdetection3d/   third-party MMDetection3D source and runtime artifacts
├── integrations/    thin framework adapters, custom components, and configs
├── routing/         LRProxy router source
├── reno/            maintained task-aware coordinate restorer and checkpoint
├── GPCC/            G-PCC measurement and rate-aggregation utilities
├── extension/       tmc3, pc_error, codec wrappers, and configs
├── scripts/         maintained project entry points grouped by function
├── experiment_results/     task-loss/BPP labels and final RD-curve artifacts
├── outputs/         complexity benchmark results
├── data_utils/      shared point-cloud and sparse-tensor utilities
├── .gitignore
└── README.md
```

Project scripts are organized as follows:

```text
scripts/
├── benchmarks/       latency, memory, and complexity measurements
├── curve_tools/      AP/BPP parsing, merging, evaluation, and plotting
├── data_preparation/ KITTI camera-FOV dataset preparation
├── evaluation/       standalone evaluation entry points
├── label_generation/ task-loss label export and shard merging
├── pipelines/        resumable end-to-end experiment orchestration
└── training/         detector, router, and geometry-restorer training
```

## KITTI Detection Assets

### Data

```text
OpenPCDet/data/kitti/       original KITTI data
OpenPCDet/data/kitti_fov/   camera-FOV-only geometry used for compression
```

The maintained FOV split contains 3,712 training frames and 3,769 validation
frames. To create or refresh the FOV copy without modifying the source KITTI
data:

```bash
cd /public/DATA/sm/RACO-LPCC
bash scripts/data_preparation/prepare_kitti_fov.sh
```

### Detector Configurations and Weights

```text
integrations/openpcdet/configs/kitti_models/pv_rcnn_fov_geometry.yaml
integrations/openpcdet/configs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml
integrations/openpcdet/configs/kitti_models/pointpillar_fov_geometry.yaml
integrations/openpcdet/configs/kitti_models/second_fov_geometry.yaml
integrations/openpcdet/configs/kitti_models/pointrcnn_fov_geometry.yaml

OpenPCDet/tools/ckpt/model_non_reflectance.pth
OpenPCDet/tools/ckpt/pointpillar_7728.pth
OpenPCDet/tools/ckpt/second_7862.pth
OpenPCDet/tools/ckpt/pointrcnn_7870.pth
```

PV-RCNN is the source task model for router training. PointPillars, SECOND,
and PointRCNN are retained only for zero-shot same-task router transfer. The
discarded Range View result is not part of the maintained comparison.

## KITTI Six-Scale Data Generation

### Training-Set G-PCC BPP

The resumable six-scale encoder entry point is:

```bash
cd /public/DATA/sm/RACO-LPCC
bash scripts/pipelines/run_kitti_train_gpcc_bpp_6scales.sh
```

Canonical retained output:

```text
experiment_results/kitti_detection_gpcc_training_bpp_6scales/gpcc_train_details.csv
```

### Training/Validation PV-RCNN Loss Labels

The general sharded exporter is:

```bash
cd /public/DATA/sm/RACO-LPCC
bash scripts/label_generation/generate_kitti_pvrcnn_detection_loss_labels.sh
```

It exports six absolute PV-RCNN total losses and also retains deltas relative
to 64 mm for inspection. Current router training uses the six absolute loss
columns.

Canonical retained labels are:

```text
experiment_results/kitti_detection_loss_labels_6scales/train_detection_loss_sensitivity.csv
experiment_results/kitti_detection_loss_labels_6scales/val_detection_loss_sensitivity.csv
```

### Fixed G-PCC Validation Baseline

```bash
cd /public/DATA/sm/RACO-LPCC
bash scripts/pipelines/run_kitti_pvrcnn_gpcc_baseline_curve.sh
```

Canonical retained validation data:

```text
experiment_results/kitti_detection_gpcc_val_bpp_6scales/gpcc_val_details.csv
experiment_results/kitti_detection_gpcc_val_bpp_6scales/baseline_ap.csv
experiment_results/kitti_detection_gpcc_val_bpp_6scales/baseline_gpcc_curve.csv
```

## LRProxy Router Source and Checkpoints

Maintained source:

```text
routing/lrproxy/
```

Important modules:

```text
gpu_voxelizer.py
lrproxy_base.py
lrproxy.py
train_kitti_lrproxy_router_ddp.py
export_kitti_lrproxy_router_labels.py
train_semantickitti_lrproxy_router_ddp.py
export_semantickitti_lrproxy_router_labels.py
eval_semantickitti_lrproxy_adaptive.py
```

Current best checkpoints:

```text
# KITTI PV-RCNN loss + G-PCC BPP
OpenPCDet/tools/router_work_dirs/
  lrproxy_kitti_pvrcnn_alltrain_trainloss_fullbpp_ddp7_20260829/best.pth

# KITTI residual-restoration-aware loss + G-PCC BPP
OpenPCDet/tools/router_work_dirs/
  lrproxy_kitti_pvrcnn_residual_scratch_q128dist100_alltrain_20260901/best.pth

# SemanticKITTI MinkUNet loss + G-PCC BPP
OpenPCDet/tools/router_work_dirs/
  monotonic_lrproxy_semantickitti_alltrain_trainloss_ddp7_20260829/best.pth
```

The KITTI zero-shot PV-RCNN/PointPillars/SECOND/PointRCNN labels and curves are
stored in:

```text
experiment_results/kitti_detection_lrproxy_pvrcnn_zero_shot_20260829/
```

## Task-Aware Geometry Restoration

Maintained source and best checkpoint:

```text
reno/current_q_ones_coordinate_v1_20260831/
reno/current_q_ones_coordinate_runs_scratch_q128dist100_lr1e4_5ep_20260901/
  best_train_loss.pth
```

The exact final training launcher is retained at:

```text
scripts/training/train_kitti_geometry_restorer_scratch_q128_chamfer100.sh
```

It trains the five restored levels and applies a 100x Chamfer multiplier at
128 mm. It writes to the existing canonical checkpoint directory; change the
output path before intentionally starting an independent rerun.

Fixed-scale mAP/PSNR results:

```text
experiment_results/gpcc_current_q_ones_scratch_q128dist100_20260901/
```

Residual-aware router pipeline and result:

```text
scripts/pipelines/run_kitti_gpcc_residual_lrproxy_router.sh
experiment_results/gpcc_current_q_ones_scratch_q128dist100_lrproxy_router_20260901/
```

The completed residual result retains both sets of per-operating-point mixed
detection PKLs. They must not be deleted: the original fixed per-scale
residual detection PKLs for five levels are no longer present.

The pipeline is resumable against its existing completion markers. A clean
rerun after deleting state requires regenerating the missing fixed per-scale
detector PKLs first.

Primary comparison artifacts:

```text
experiment_results/gpcc_current_q_ones_scratch_q128dist100_lrproxy_router_20260901/
  comparison/four_curve_map_bpp.png
  comparison/five_curve_map_bpp.png
  comparison/bd_rate_vs_fixed_gpcc.csv
```

The true-loss oracle under
`experiment_results/gpcc_reno_true_loss_oracle_no64_20260831/` is retained only as a
diagnostic upper bound; it is not part of the primary test figure.

## SemanticKITTI

Project-owned MinkUNet datasets, metrics, tools, and configs are under
`integrations/mmdetection3d/`; the third-party implementation and retained
pretrained checkpoint remain under:

```text
mmdetection3d/
mmdetection3d/checkpoints/
  minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth
```

The current LRproxy router was trained on all 19,130 merged training
frames and selected at epoch 27 by minimum full-training regression loss.
Sequence 08 is used only for final evaluation.

Current baseline and routed curves:

```text
mmdetection3d/work_dirs/semantickitti_gpcc_fixed_baseline_6scales/
  gpcc_fixed_step_miou_bpp.csv
mmdetection3d/work_dirs/monotonic_lrproxy_semantickitti_seq08_20260829/
  lrproxy_guided_miou_bpp.csv
```

Experiments for nuScenes, ModelNet40, ShapeNet55, and SUN RGB-D are maintained
on the second server and are not represented by paths in this checkout.

## Complexity Benchmarks

Maintained benchmark entry points:

```text
scripts/benchmarks/benchmark_kitti_geometry_restorer.py
scripts/benchmarks/benchmark_kitti_geometry_restorer_inference_paths.py
scripts/benchmarks/benchmark_kitti_detector_one_frame.py
scripts/benchmarks/benchmark_routing_components_one_frame.py
scripts/benchmarks/benchmark_semantickitti_minkunet_forward.py
```

Each CUDA timing benchmark performs warmup and synchronizes CUDA around timed
regions. Use `--help` on an entry point for required frame, checkpoint, and
output arguments.

Retained measurements:

```text
outputs/complexity_benchmarks/routing_codec_and_minkunet_20260826/
outputs/complexity_benchmarks/machine_task_models_20260828/
```

Core-network latency excludes preprocessing. End-to-end latency includes the
preprocessing explicitly documented in each JSON. Peak `allocated` and
`reserved` CUDA memory use different allocator accounting and must not be
interchanged.

## Result and Cache Policy

- Reuse measured G-PCC BPP whenever only the router or coordinate restorer
  changes.
- The coordinate restorer adds no bitstream payload.
- Keep merged task-loss CSVs, per-frame BPP CSVs, best checkpoints, final
  curves, completion markers, and the surviving mixed detection PKLs.
- Shards may be removed only after frame-ID coverage is proven identical to
  the merged CSV.
- Do not use KITTI validation data to select lambdas or epochs.
- Do not overwrite an existing experiment directory; use a new semantic name
  and date for a new run.
- Historical logs/checkpoint metadata may contain old absolute paths. Those
  strings are provenance records, not active code references.

## Quick Integrity Checks

```bash
cd /public/DATA/sm/RACO-LPCC

bash -n scripts/pipelines/run_kitti_pvrcnn_gpcc_baseline_curve.sh
bash -n scripts/pipelines/run_kitti_gpcc_residual_lrproxy_router.sh

/home/sm/miniconda3/envs/SparsePCGC/bin/python \
  routing/lrproxy/train_kitti_lrproxy_router_ddp.py \
  --help
```

Before launching a long job, check `gpustat`, existing processes, completion
markers, and whether the target output directory already exists.
