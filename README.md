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

### Train Detector With a Custom KITTI Train/Val Split

Split the KITTI training set by count, using the first part for training. The
validation set is the held-out second part plus the original KITTI `val` split.
The default train:held-out ratio below is `5:1`.

Prepare split files, filtered `kitti_infos_*.pkl`, GT database, and a generated
PV-RCNN config:

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python train_kitti_split_detector.py \
  --ratio 5:1
```

To also train PV-RCNN from scratch and evaluate saved checkpoints on the
combined validation split:

```bash
CUDA_VISIBLE_DEVICES=1 \
python train_kitti_split_detector.py \
  --ratio 5:1 \
  --epochs 80 \
  --batch-size 4 \
  --run-train
```

The script writes `best_val_checkpoint.txt` under the OpenPCDet output
directory. By default, the best checkpoint is selected by the mean of Car,
Pedestrian, and Cyclist moderate 3D AP on the combined validation split. Add
`--no-extra-val-splits` if you only want to validate on the held-out training
subset.

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
FG_WEIGHT=20.0 \
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
`VAL_RATIO`, `FG_WEIGHT`, and `WORK_DIR`. `FG_WEIGHT` defaults to `20.0` to
bias the segmentation model toward foreground recall. Increase it further if
missing foreground points is more harmful than adding background false
positives.

Example recall-biased training command:

```bash
CUDA_VISIBLE_DEVICES=6 \
FG_WEIGHT=20.0 \
./run_kitti_box_seg_train.sh
```

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

## Compute Oracle Router Upper Bound

This diagnostic bypasses the learned router proxy. It uses the true per-frame
AP sensitivity matrix from `new_split.py` and the existing Split-GPCC
per-frame bit counts to choose an oracle label for each frame. The default
objective is Car AP-bpp:

```text
label_i = argmin_l  CarDrop_i(l) + lambda * frame_bpp_i(l)
```

Sweep `lambda` to produce an oracle AP-bpp curve. If this oracle curve still
does not beat fixed Split-GPCC, the current per-frame routing formulation is
not strong enough. If it beats Split-GPCC, the learned proxy is the bottleneck.

First generate the true per-frame AP sensitivity matrix on the same val split
and the same combo `result.pkl` files used by router evaluation:

```bash
cd /public/DATA/sm/RACO-LPCC/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=6 \
/home/sm/miniconda3/envs/SparsePCGC/bin/python new_split.py \
  --cfg_file cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
  --split_file ../data/kitti_fov/ImageSets/val.txt \
  --eval_dir ../output/kitti_models/pv_rcnn_fov_geometry/juqp_model_w_juqp_val/eval/epoch_no_number/val/default \
  --out_csv ../../point_pairs/oracle_router_fov/val_ap_sensitivity.csv \
  --workers 16 \
  --quant_map '1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048'
```

Then compute the oracle router curve:

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_oracle_router_curve.py \
  --cfg_file OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
  --eval_dir OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/juqp_model_w_juqp_val/eval/epoch_no_number/val/default \
  --ap_csv point_pairs/oracle_router_fov/val_ap_sensitivity.csv \
  --split_details_csv point_pairs/split_gpcc_fov/gpcc/split_all_details.csv \
  --quant_map '1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048' \
  --objective Car \
  --out_dir point_pairs/oracle_router_fov
```

Default outputs:

```text
point_pairs/oracle_router_fov/oracle_router_curve.csv
point_pairs/oracle_router_fov/oracle_average_results.csv
point_pairs/oracle_router_fov/oracle_all_details.csv
point_pairs/oracle_router_fov/oracle_rate_*_labels.csv
```

## Train JUQP Router Proxy Model

Use the one-shot script to build JUQP labels on the original KITTI train split
with `model_w_juqp.pth`, then train and calibrate the router proxy model:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=6 ./run_juqp_train_router.sh
```

The script uses the original KITTI `train.txt` split for JUQP label generation.
For router proxy training, it further splits that original train set into a
train subset and a validation subset; the validation subset is used for cost
head calibration.

Default detection weight:

```text
OpenPCDet/tools/ckpt/model_w_juqp.pth
```

Default quantization map:

```text
1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048
```

If detection intermediate results or AP matrices have already been generated,
skip the expensive earlier stages:

```bash
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 CUDA_VISIBLE_DEVICES=6 ./run_juqp_train_router.sh
```

To only train the router from existing JUQP labels and AP CSV:

```bash
RUN_TEST_SPLIT=0 \
RUN_NEW_SPLIT=0 \
RUN_JUQP_LABELS=0 \
CUDA_VISIBLE_DEVICES=6 \
./run_juqp_train_router.sh
```

Default outputs:

```text
OpenPCDet/tools/split_AP_train_model_w_juqp.csv
OpenPCDet/tools/juqp_train_labels_model_w_juqp/
OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp/
```

### Fine-tune Router With Larger Car Weight

If the learned router AP-bpp curve is mainly worse on Car while Pedestrian and
Cyclist improve, fine-tune the current router from its existing checkpoint and
increase the Car AP-drop loss weight. This reuses the AP sensitivity CSV and
the train/validation split already produced by `./run_juqp_train_router.sh`.
The command fine-tunes the full router proxy for 40 epochs, then
`--calibrate_cost` trains a new calibration layer from the fine-tuned
`best.pth` and saves `calibration.pth` in the same output directory.
This variant also allows signed, non-monotonic AP-drop costs, because lower
geometry precision can occasionally improve AP on some frames and forcing
`cost >= 0` or monotonic cost can hurt Car prediction.

```bash
cd /public/DATA/sm/RACO-LPCC/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=1 \
/home/sm/miniconda3/envs/SparsePCGC/bin/python train_cost_proxy.py \
  --velodyne_dir ../data/kitti/training/velodyne \
  --train_split router_splits/cost_proxy_model_w_juqp/train_from_original_train.txt \
  --ap_csv split_AP_train_model_w_juqp.csv \
  --val_split router_splits/cost_proxy_model_w_juqp/val_from_original_train.txt \
  --val_ap_csv split_AP_train_model_w_juqp.csv \
  --thresholds '0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075' \
  --out_dir router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft \
  --pretrained_ckpt router_work_dirs/ckpts/best_nocalib.pth \
  --epochs 40 \
  --batch_size 8 \
  --workers 4 \
  --voxel_size 0.16 0.16 0.16 \
  --point_cloud_range 0 -40 -3 70.4 40 1 \
  --max_voxels 50000 \
  --feat_dim 256 \
  --ap_drop_scale 100 \
  --signed_ap_drop \
  --allow_negative_cost \
  --no_monotonic_cost \
  --lambda_threshold 0.1 \
  --ap_weights 30.0 1.0 1.0 \
  --lr 1e-4 \
  --jitter_std 0.005 \
  --weight_decay 5e-4 \
  --device cuda \
  --calibrate_cost \
  --calibration_epochs 20
```

Then evaluate the router curve with the fine-tuned checkpoint:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=1 \
RUN_TEST_SPLIT=0 \
ROUTER_CKPT=OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/best.pth \
ROUTER_CALIBRATION=OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/calibration.pth \
THRESHOLDS='0,0,0;0.0001,0.01,0.02;0.0002,0.02,0.035;0.0003,0.03,0.045;0.0004,0.04,0.06;0.0005,0.05,0.075' \
./run_router_gpcc_curve.sh
```

## Train Super Resolution Module

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u GPCC/train_sparse_sr.py \
  --device cuda \
  --epochs 80 \
  --eval_interval 10 \
  --batch_size 2 \
  --workers 4 \
  --direct_coarse_quant \
  --work_dir GPCC/work_dirs/sparse_sr_direct
```

With `--direct_coarse_quant`, training uses `round(coords * scale)` as the
coarse input lattice and supervises the 8 child positions implied by
`coarse * 2 + offset`, matching the full-frame SR test path below.

## Test Super Resolution Module

The first frame in the current KITTI FOV val split is `000001`, so the frame
argument can be passed as `000001.bin`. The command below evaluates the trained
super-resolution checkpoint on the full frame at the same 8 quantization scales
used by `GPCC/train_sparse_sr.py`; it does not require foreground/background
segmentation labels.

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=3 \
python -u GPCC/eval_sr_frame_psnr_bpp.py \
  --device cuda \
  --frame 000001.bin \
  --ckpt GPCC/work_dirs/sparse_sr_direct/latest.pth \
  --selection oracle_count \
  --metadata_bits 32 \
  --out_dir GPCC/outputs_sr_eval/000001_latest
```

`--selection oracle_count` simulates the decoder using the SR model with the
true next-resolution full-frame point count carried in the bitstream. The
reported SR bitrate therefore adds `--metadata_bits` to each frame/scale result
to record that count. If the checkpoint was trained into another directory,
for example `GPCC/work_dirs/sparse_r`, change `--ckpt` to that directory's
`latest.pth` or `best.pth`.

Outputs:

```text
GPCC/outputs_sr_eval/000001_latest/sr_psnr_bpp.csv
GPCC/outputs_sr_eval/000001_latest/gpcc_psnr_bpp.csv
GPCC/outputs_sr_eval/000001_latest/d1_psnr_bpp_curve.png
GPCC/outputs_sr_eval/000001_latest/d2_psnr_bpp_curve.png
```

The two CSV files contain D1/D2 PSNR and bpp for SR and non-SR GPCC baseline
at all 8 quantization scales. The two PNG files plot SR and baseline curves
on the same figure for D1 and D2 separately.


## Obtain AP-bpp curve point pairs with the JUQP router proxy

After `./run_juqp_train_router.sh` finishes, use the trained router proxy to
predict AP-drop costs on the evaluation split, derive one adaptive JUQP label
CSV per threshold triple, and then evaluate compression/AP curve points. This
stage runs the detector once for each fixed quantization combo on the val split
to save `combo_*/result.pkl`, then computes router-assisted AP by selecting the
per-frame annotation from the combo chosen by the router. Compression
`bpp`/time are still aggregated from the existing Split-GPCC
`split_all_details.csv`; no G-PCC compression is rerun.

The detector combo `result.pkl` files should be generated with the same
checkpoint you want to report. To match the router training setup above, the
default is `model_w_juqp.pth`. `SPLIT_FILE`, `CFG_FILE`, `SPLIT_EVAL_DIR`, and
`SPLIT_DETAILS_CSV` must all refer to the same val frame split.

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=3 \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
ROUTER_CKPT=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/best.pth \
ROUTER_CALIBRATION=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/calibration.pth \
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
DET_CKPT=ckpt/model_w_juqp.pth \
DET_EXTRA_TAG=juqp_model_w_juqp_val \
THRESHOLDS='0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.045,0.05,0.075' \
SPLIT_DETAILS_CSV=point_pairs/split_gpcc_fov/gpcc/split_all_details.csv \
./run_router_gpcc_curve.sh
```

If the detector combo `result.pkl` files already exist, skip that detector
stage and reuse them:

```bash
RUN_TEST_SPLIT=0 \
SPLIT_EVAL_DIR=OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/juqp_model_w_juqp_val/eval/epoch_no_number/val/default \
./run_router_gpcc_curve.sh
```

To try the debt-aware routing policy without retraining the router proxy, keep
the same checkpoint and reuse the fixed combo `result.pkl` files. This policy
allows a frame to exceed the Car threshold by a small amount when the predicted
AP-drop increase buys enough bpp reduction, records the overshoot as debt, and
tightens later Car thresholds until the debt is repaid:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=3 \
RUN_TEST_SPLIT=0 \
SELECTION_POLICY=debt \
DEBT_TARGET=car \
DEBT_ALPHA=1.5 \
DEBT_BETA=1.0 \
DEBT_MAX_EXTRA=0.0003 \
DEBT_MIN_THRESHOLD_RATIO=0.7 \
DEBT_MIN_SAVING_PER_COST=500 \
BPP_ESTIMATE=mean \
THRESHOLDS='0,0,0;0.0001,0.01,0.02;0.0002,0.02,0.035;0.0003,0.03,0.045;0.0004,0.04,0.06;0.0005,0.05,0.075' \
ROUTER_CKPT=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/best.pth \
ROUTER_CALIBRATION=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/calibration.pth \
SPLIT_EVAL_DIR=OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/juqp_model_w_juqp_val/eval/epoch_no_number/val/default \
SPLIT_DETAILS_CSV=point_pairs/split_gpcc_fov/gpcc/split_all_details.csv \
./run_router_gpcc_curve.sh
```

Debt-policy label CSVs add `hard_label`, `debt_extra`, `debt_bpp_saving`,
`debt_after`, and `effective_threshold` columns for debugging. Increase
`DEBT_MAX_EXTRA` or lower `DEBT_MIN_SAVING_PER_COST` to make routing more
aggressive; lower `DEBT_MAX_EXTRA` or raise `DEBT_MIN_SAVING_PER_COST` to make
it closer to the hard-threshold policy. `BPP_ESTIMATE=mean` uses one average
bpp per quantization label for all frames, which better simulates deployment
than using per-frame compressed bpp values.

To generate route points by directly minimizing a weighted AP-drop/bpp
objective, use the Lagrangian policy. This does not use `THRESHOLDS` for label
selection; each lambda value produces one curve point:

```text
label_i = argmin_l weighted_predicted_AP_drop_i(l) + lambda * frame_bpp_i(l)
```

For a Car-prioritized but three-class-aware curve, include Pedestrian and
Cyclist with smaller AP-drop weights:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=3 \
RUN_TEST_SPLIT=0 \
SELECTION_POLICY=lagrangian \
BPP_ESTIMATE=mean \
LAGRANGE_CLASS_WEIGHTS='1,0,0' \
LAGRANGE_LAMBDAS='0,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.005,0.01' \
LAGRANGE_MAX_LABELS='1,4,4,4,4,4,4,5,6' \
ROUTER_CKPT=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/best.pth \
ROUTER_CALIBRATION=/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/router_work_dirs/cost_proxy_model_w_juqp_car_signed_ft/calibration.pth \
SPLIT_EVAL_DIR=OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/juqp_model_w_juqp_val/eval/epoch_no_number/val/default \
SPLIT_DETAILS_CSV=point_pairs/split_gpcc_fov/gpcc/split_all_details.csv \
./run_router_gpcc_curve.sh
```

The command writes to the default router paths and can overwrite the previous
router curve CSVs. Lagrangian label CSVs add `lagrange_lambda`,
`lagrange_score`, `weighted_ap_drop`, and `label_bpp` columns for debugging.
If all points are too high-bpp, increase the lambda range; if all points are
too low-bpp, decrease it. With `BPP_ESTIMATE=mean`, `label_bpp` is the same
average bpp for a given quantization label on every frame. `LAGRANGE_MAX_LABELS`
sets the maximum allowed quantization label for each lambda point, which keeps
middle-bpp points from jumping directly to the lowest-rate labels.

For each threshold triple, the router predicts six AP-drop cost heads
corresponding to `L1` through `L6`. The selected label is the largest label
whose predicted `Car/Pedestrian/Cyclist` costs are all within that threshold;
label `0` is used as the fallback/highest-quality combination. The label maps
back to `QUANT_MAP` in order, so the default label mapping is:

When `SELECTION_POLICY=debt`, this hard-threshold label is still recorded as
`hard_label`, but the final `jucp_label` may be a lower-rate label if its Car
overshoot fits the dynamic debt budget.

```text
L0 -> 1/64,1/64
L1 -> 1/64,3/256
L2 -> 1/64,2/256
L3 -> 1/64,1/256
L4 -> 1/64,1.5/512
L5 -> 1/64,1/512
L6 -> 1/64,1/2048
```

Default outputs:

```text
point_pairs/router_gpcc_fov/labels/router_costs.csv
point_pairs/router_gpcc_fov/labels/router_rate_*.csv
point_pairs/router_gpcc_fov/gpcc/router_all_details.csv
point_pairs/router_gpcc_fov/gpcc/router_average_results.csv
point_pairs/router_gpcc_fov/router_ap.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
```

`router_costs.csv` records the router-predicted AP-drop costs for every frame
and level. Each `router_rate_*.csv` records the per-frame selected
`jucp_label` for one threshold/curve point. `router_gpcc_curve.csv` is the
final point-pair table with `bpp`, `enc_time`, `dec_time`, and the three KITTI
moderate 3D AP_R40 values.

If labels or AP have already been generated, skip stages as needed:

```bash
RUN_EXPORT=0 RUN_AP=0 ./run_router_gpcc_curve.sh
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

## Obtain AP-bpp curve point pairs of Split-GPCC method

After training the geometry-only foreground/background segmentation model, run
Split-GPCC on the same FOV KITTI split used by the baseline curve. The pipeline
is the same as `./run_gpcc_baseline_curve.sh`, except that each frame is first
split into foreground/background by the segmentation checkpoint and the split
mask is cached. Later runs reuse the cached split file and skip segmentation.

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=1 \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
SEG_CFG=mmdetection3d/configs/minkunet/minkunet34_w32_minkowski_geometry_kitti_box_seg.py \
SEG_CKPT=mmdetection3d/work_dirs/minkunet_kitti_fov_box_seg_geometry/best_foreground_epoch_33.pth \
MASK_DIR=point_pairs/split_gpcc_fov/seg_masks \
SEG_TIME_CSV=point_pairs/split_gpcc_fov/seg_time.csv \
SEG_FG_THRESHOLD=0.35 \
SPLIT_SCALES='1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048' \
OUT_DIR=point_pairs/split_gpcc_fov \
./run_split_gpcc_curve.sh
```

The script runs these stages:

1. Evaluate OpenPCDet AP under the Split-GPCC reconstructed point clouds.
2. Generate foreground/background masks into `MASK_DIR`. For each frame, if
   `${MASK_DIR}/<frame_id>.npy` already exists, skip segmentation for that
   frame. Otherwise run the geometry-only segmentation model, save the mask, and
   append that frame's `seg_time` to `SEG_TIME_CSV`. `SEG_FG_THRESHOLD`
   controls the foreground probability threshold during mask generation; lower
   values increase foreground recall and may mark more background points as
   foreground.
3. Compress foreground and background independently with G-PCC, record
   `fg_enc_time`, `bg_enc_time`, `enc_time`, `dec_time`, `bits`, and `bpp`.
4. Merge AP and Split-GPCC metrics into one AP-bpp/AP-time curve CSV.

Default split quantization combinations should match the detection-side split
evaluation. The foreground precision is fixed at `1/64`, while the background
precision is swept from high to low so high-bpp AP can be inspected early:

```text
1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;
1/64,1.5/512;1/64,1/512;1/64,1/2048
```

Default outputs:

```text
point_pairs/split_gpcc_fov/split_ap.csv
point_pairs/split_gpcc_fov/seg_masks/<frame_id>.npy
point_pairs/split_gpcc_fov/seg_time.csv
point_pairs/split_gpcc_fov/gpcc/split_all_details.csv
point_pairs/split_gpcc_fov/gpcc/split_average_results.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
```

For time comparison, the Split-GPCC encoding time should include segmentation:

```text
total_enc_time = seg_time + fg_enc_time + bg_enc_time
```

If all masks already exist and you only want to recompute Split-GPCC rates and
merge:

```bash
RUN_SEG=0 \
MASK_DIR=point_pairs/split_gpcc_fov/seg_masks \
RUN_AP=0 \
AP_CSV=point_pairs/split_gpcc_fov/split_ap.csv \
PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python \
./run_split_gpcc_curve.sh
```

Common overrides:

```bash
SPLIT_SCALES='1/64,1/64;1/64,3/256;1/64,2/256;1/64,1/256;1/64,1.5/512;1/64,1/512;1/64,1/2048' \
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
CKPT=ckpt/model_non_reflectance.pth \
KITTI_ROOT=OpenPCDet/data/kitti_fov \
SPLIT_FILE=OpenPCDet/data/kitti_fov/ImageSets/val.txt \
GPCC_CFG=extention/kitti.cfg \
OUT_DIR=point_pairs/split_gpcc_fov \
SEG_FG_THRESHOLD=0.35 \
CUDA_VISIBLE_DEVICES=1 \
./run_split_gpcc_curve.sh
```

## Obtain PSNR-bpp curve point pairs of 3 methods

PSNR-bpp points are computed separately from plotting. The script below does
not rerun G-PCC compression. It reads the existing bpp values from the AP/bpp
curve CSVs, simulates the same coordinate quantization used by the AP
evaluation scripts, writes temporary positive-integer PLY files, calls
`extention/pc_error_d` through `extention/pc_error_geo.py`, then deletes the
temporary PLY files.

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_psnr_bpp_curves.py
```

This runs all selected methods on the whole split, so it can take a long time:
each rate point calls `pc_error_d` once per frame. For a quick smoke test, run
only the first few frames:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_psnr_bpp_curves.py \
  --methods baseline \
  --max_frames 3
```

Default inputs:

```text
OpenPCDet/data/kitti_fov/training/velodyne/
OpenPCDet/data/kitti_fov/ImageSets/val.txt
point_pairs/baseline_fov/baseline_gpcc_curve.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
point_pairs/split_gpcc_fov/seg_masks/<frame_id>.npy
point_pairs/router_gpcc_fov/gpcc/router_all_details.csv
```

Default outputs:

```text
point_pairs/psnr_bpp/baseline_psnr_bpp_curve.csv
point_pairs/psnr_bpp/split_psnr_bpp_curve.csv
point_pairs/psnr_bpp/router_psnr_bpp_curve.csv
point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv
```

The output CSVs contain `bpp`, `psnr_p2point`, and `mse_p2point`. Add
`--keep_details` to also save per-frame PSNR rows:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_psnr_bpp_curves.py --keep_details
```

Common overrides:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_psnr_bpp_curves.py \
  --testdata OpenPCDet/data/kitti_fov/training/velodyne \
  --split_file OpenPCDet/data/kitti_fov/ImageSets/val.txt \
  --mask_dir point_pairs/split_gpcc_fov/seg_masks \
  --baseline_curve_csv point_pairs/baseline_fov/baseline_gpcc_curve.csv \
  --split_curve_csv point_pairs/split_gpcc_fov/split_gpcc_curve.csv \
  --router_curve_csv point_pairs/router_gpcc_fov/router_gpcc_curve.csv \
  --router_details_csv point_pairs/router_gpcc_fov/gpcc/router_all_details.csv \
  --out_dir point_pairs/psnr_bpp \
  --tmp_dir point_pairs/psnr_tmp \
  --resolution 80000
```


## Plot curve figures

### AP curves

After baseline, Split-GPCC, and JUQP Router curve CSV files are generated, run:

```bash
cd /public/DATA/sm/RACO-LPCC

python plot_all_curves.py
```

The script reads these files by default:

```text
point_pairs/baseline_fov/baseline_gpcc_curve.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
```

If the JUQP Router CSV does not exist yet, the script skips that optional curve
and still plots the baseline and Split-GPCC curves.

It writes all figures to `plots/`:

```text
plots/ap_bpp_car.png
plots/ap_bpp_pedestrian.png
plots/ap_bpp_cyclist.png
plots/ap_enctime_car.png
plots/ap_enctime_pedestrian.png
plots/ap_enctime_cyclist.png
plots/ap_dectime_car.png
plots/ap_dectime_pedestrian.png
plots/ap_dectime_cyclist.png
```

To override input or output paths:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python plot_all_curves.py \
  --baseline_csv point_pairs/baseline_fov/baseline_gpcc_curve.csv \
  --split_csv point_pairs/split_gpcc_fov/split_gpcc_curve.csv \
  --juqp_csv point_pairs/router_gpcc_fov/router_gpcc_curve.csv \
  --out_dir plots
```

### PSNR curves

After `all_methods_psnr_bpp_curve.csv` is generated, draw the PSNR-bpp figure:

```bash
python plot_psnr_bpp_curves.py
```

Default figure output:

```text
plots/psnr_bpp.png
```

To override paths:

```bash
python plot_psnr_bpp_curves.py \
  --csv point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv \
  --out_dir plots \
  --formats png,pdf
```


## Compute BD-rate

After the baseline, Split-GPCC, and JUQP Router AP-bpp curve CSV files are
ready, compute class-wise BD-rate with:

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_bdrate.py
```

The script reads these default AP-bpp point-pair CSVs:

```text
point_pairs/baseline_fov/baseline_gpcc_curve.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
```

It reports BD-rate for the three target classes (`Car`, `Pedestrian`,
`Cyclist`) for:

```text
JUQP Router vs Baseline G-PCC
JUQP Router vs Split-GPCC
Split-GPCC vs Baseline G-PCC
```

Negative values mean the compared method saves bitrate relative to the
reference method at the same AP. To override paths and save a CSV table:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python compute_bdrate.py \
  --baseline_csv point_pairs/baseline_fov/baseline_gpcc_curve.csv \
  --split_csv point_pairs/split_gpcc_fov/split_gpcc_curve.csv \
  --juqp_csv point_pairs/router_gpcc_fov/router_gpcc_curve.csv \
  --out_csv plots/bdrate_ap_bpp.csv
```
