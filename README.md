# RACO-LPCC

This project studies geometry compression for LiDAR point clouds. The semantic
segmentation model source is based on
https://github.com/open-mmlab/mmdetection3d, and the object detection model
source is based on https://github.com/open-mmlab/OpenPCDet.

## Prepare KITTI Dataset and Related Model Checkpoints

todo

## Prepare Camera-FOV-only KITTI

The OpenPCDet KITTI configs already apply `FOV_POINTS_ONLY=True` before model
inference. To make compression use exactly the same points, create a separate
KITTI root whose `.bin` files physically exclude all points outside the front
camera image:

```bash
cd /public/DATA/sm/RACO-LPCC
./prepare_kitti_fov.sh
```

The source dataset is not modified. The cropped dataset is written to:

```text
OpenPCDet/data/kitti_fov/
```

Calibration, images, labels, split files, info files, and the ground-truth
database are linked from the original dataset. Only `training/velodyne` (and
`testing/velodyne`, when present) is regenerated. Per-frame point counts are
recorded in `fov_crop_stats.csv`.

Use these detection configs:

```text
cfgs/kitti_models/pv_rcnn_fov_geometry.yaml
cfgs/kitti_models/pv_rcnn_train_as_test_fov_geometry.yaml
```

The baseline G-PCC and JUQP shell scripts use the FOV-only data/configs by
default. Existing segmentation masks generated for full point clouds must be
regenerated against `kitti_fov`, because their point counts no longer match.


## Train Geometry-only Models

The original segmentation and detection models use LiDAR reflectance
(`x, y, z, intensity`) as input. This repo now provides geometry-only variants
that use only `x, y, z` as network input.

The raw KITTI / SemanticKITTI binary files can still contain the intensity
column. The geometry-only configs read the file with `load_dim=4` or
`src_feature_list=['x', 'y', 'z', 'intensity']`, but only pass `x, y, z` into
the model.

### Fine-tune PV-RCNN Without Reflectance

```bash
cd /public/DATA/sm/RACO-LPCC/OpenPCDet/tools

python train_geometry_only.py \
  --pretrained_model ckpt/model_w_reflectance.pth \
  --batch_size 4 \
  --epochs 20
```

Default geometry-only config:

```text
cfgs/kitti_models/pv_rcnn_geometry.yaml
```

Train-as-test geometry-only config:

```text
cfgs/kitti_models/pv_rcnn_train_as_test_geometry.yaml
```

The loader is compatible with the old 4-channel checkpoint. For the first input
convolution, it keeps the pretrained `x, y, z` weights and drops the reflectance
channel weight. Other compatible weights are loaded normally.

### Fine-tune MinkUNet Without Reflectance(replaced by next part)

```bash
cd /public/DATA/sm/RACO-LPCC/mmdetection3d

python tools/train_geometry_only.py \
  --pretrained ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth
```

Default geometry-only config:

```text
configs/minkunet/minkunet34_w32_minkowski_geometry_8xb2-laser-polar-mix-3x_semantickitti.py
```

This training entry also supports loading the old 4-channel checkpoint and
adapting the first input convolution to 3 channels.

### Train KITTI Box-supervised Foreground Segmentation

Generate foreground/background point labels from KITTI 3D boxes, split 10% of
the KITTI training split for validation, and train a geometry-only MinkUNet:

```bash
cd /public/DATA/sm/RACO-LPCC
CUDA_VISIBLE_DEVICES=6 \
./run_kitti_box_seg_train.sh
```

Only points inside the front-camera field of view are supervised by default.
Points inside Car, Pedestrian, or Cyclist boxes are foreground, points outside
the boxes are background, and points outside the image field of view are
ignored. The checkpoint with the highest validation foreground IoU is saved
under:

```text
mmdetection3d/work_dirs/minkunet_kitti_fov_box_seg_geometry/
```

Optional environment variables include `KITTI_ROOT`, `PYTHON_BIN`,
`VAL_RATIO`, `FG_WEIGHT`, and `WORK_DIR`.

FOV-only labels and indexes are kept separate from older full-frame labels:

```text
mmdetection3d/data/kitti_fov_box_seg/
```

The labels are cached as one `box_seg_labels/<frame_id>.label` file per point
cloud, with train/validation indexes in
`kitti_box_seg_infos_{train,val}.pkl`. Later runs reuse this cache and do not
regenerate labels. To rebuild them explicitly:

```bash
FORCE_REGENERATE_LABELS=1 ./run_kitti_box_seg_train.sh
```

## Obtain AP-bpp curve point pairs of the baseline method(G-PCC)

Use the one-shot baseline script:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
./run_gpcc_baseline_curve.sh
```

The script does three things:

1. Runs OpenPCDet detection evaluation under different whole-frame
   quantization scales and extracts the 3D AP_R40 values.
2. Runs baseline whole-frame G-PCC on the same KITTI split and records
   `bpp`, `enc_time`, and `dec_time`.
3. Merges AP and G-PCC metrics into one CSV for plotting AP-bpp and AP-time
   curves.

The AP stage shows a `Baseline AP scales` progress bar over quantization
scales. The G-PCC stage shows a `Baseline G-PCC` progress bar over all
frame-scale jobs.

For each frame, the G-PCC detail CSV records `num_points` and `bits`. The final
baseline `bpp` is computed as `sum(bits) / sum(num_points)` for all frames at
the same quantization scale.

Default quantization scales:

```text
1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512
```

Default outputs:

```text
point_pairs/baseline_fov/baseline_ap.csv
point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv
point_pairs/baseline_fov/baseline_gpcc_curve.csv
```

The final CSV contains:

```text
rate_id, scale, posQuantscale, bpp, enc_time, dec_time,
Car_3d_AP_R40_moderate, Pedestrian_3d_AP_R40_moderate,
Cyclist_3d_AP_R40_moderate
```

These AP columns are KITTI `3d AP_R40` values at the `moderate` difficulty
level, not confidence-threshold results. The IoU thresholds are:

```text
Car_3d_AP_R40_moderate: 3D IoU 0.70
Pedestrian_3d_AP_R40_moderate: 3D IoU 0.50
Cyclist_3d_AP_R40_moderate: 3D IoU 0.50
```

Common overrides:

```bash
SCALES='1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512' \
BATCH_SIZE=4 \
WORKERS=4 \
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
CKPT=ckpt/model_non_reflectance.pth \
OUT_DIR=point_pairs/baseline_fov \
CUDA_VISIBLE_DEVICES=1 \
./run_gpcc_baseline_curve.sh
```

If AP has already been evaluated and you only want to recompute G-PCC and merge:

```bash
RUN_AP=0 \
AP_CSV=point_pairs/baseline_fov/baseline_ap.csv \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
./run_gpcc_baseline_curve.sh
```

If G-PCC has already been measured and you only want to recompute AP and merge:

```bash
RUN_GPCC=0 \
GPCC_RESULTS_DIR=point_pairs/baseline_fov/gpcc \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
./run_gpcc_baseline_curve.sh
```














## Run JUQP With Geometry-only Detection

`juqp.sh` and `juqp_train.sh` now default to the geometry-only PV-RCNN configs.

Validation split:

```bash
cd /public/DATA/sm/RACO-LPCC
CUDA_VISIBLE_DEVICES=0 ./juqp.sh
```

Train split:

```bash
cd /public/DATA/sm/RACO-LPCC
CUDA_VISIBLE_DEVICES=0 ./juqp_train.sh
```

To temporarily use the original reflectance-based config, override `CFG_FILE`:

```bash
CFG_FILE=cfgs/kitti_models/pv_rcnn.yaml ./juqp.sh
```
