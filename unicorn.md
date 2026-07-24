# Unicorn geometry replication

This follows the RENO-style flow in `RACO-LPCC`, but uses Unicorn from `/public/DATA/sm/Unicorn`.

The script trains all three Unicorn geometry modules from scratch by importing the model classes directly:

- lossless geometry module: `Unicorn/lossless_geometry/model.py::PCCModel`
- SR module: `Unicorn/lossy_geometry/model.py::PCCModel`
- offset module: `Unicorn/lossy_geometry/model_offset.py::OffsetModel`

The trained weights are saved under:

```text
RACO-LPCC/unicorn/checkpoints/lossless/epoch_last.pth
RACO-LPCC/unicorn/checkpoints/sr/epoch_last.pth
RACO-LPCC/unicorn/checkpoints/offset/epoch_last.pth
```

The KITTI train and val samples stay as KITTI `.bin` files. The RACO-LPCC wrapper reads `.bin` directly, applies the same GPCC-style per-frame offset as RENO, and builds Minkowski sparse tensors with `TRAIN_POSQ=64.0`. The lossless module uses the same multi-scale BCE objective as `Unicorn/lossless_geometry/train.py`; its default architecture is `stage=8`, `scale=5`, `kernel_size=5`, `block_type=conv`. The wrapper does not call Unicorn's original `PCDataset`, `Trainer`, `train.py`, `train_offset.py`, or `test.py` pipeline, and it does not use earlier higher-scale decoded samples as training data.

## 0. Environment

```bash
cd /public/DATA/sm/RACO-LPCC

export UNICORN_ROOT=/public/DATA/sm/Unicorn
export PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python
export UNICORN_PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python

export KITTI_ROOT=/public/DATA/sm/RACO-LPCC/OpenPCDet/data/kitti_fov
export KITTI_VELODYNE=${KITTI_ROOT}/training/velodyne
export TRAIN_SPLIT_FILE=${KITTI_ROOT}/ImageSets/train.txt
export SPLIT_FILE=${KITTI_ROOT}/ImageSets/val.txt
```

## 1. Prepare FOV KITTI

```bash
cd /public/DATA/sm/RACO-LPCC
./prepare_kitti_fov.sh
```

## 2. Train Unicorn geometry modules

Train lossless geometry, SR, and offset together:

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=4 \
TRAIN_POSQ=64.0 \
POSQUANTSCALE_LIST="2 4 8 16 32" \
RUN_TRAIN=1 RUN_UNICORN=0 RUN_AP=0 \
./run_unicorn_lossy_geometry_curve.sh
```

Train only the lossless geometry module:

```bash
CUDA_VISIBLE_DEVICES=4 \
RUN_TRAIN=0 RUN_LOSSLESS=1 RUN_SR=0 RUN_OFFSET=0 \
RUN_UNICORN=0 RUN_AP=0 \
./run_unicorn_lossy_geometry_curve.sh
```

Train only SR or offset:

```bash
CUDA_VISIBLE_DEVICES=4 \
RUN_TRAIN=0 RUN_LOSSLESS=0 RUN_SR=1 RUN_OFFSET=0 \
RUN_UNICORN=0 RUN_AP=0 \
./run_unicorn_lossy_geometry_curve.sh

CUDA_VISIBLE_DEVICES=4 \
RUN_TRAIN=0 RUN_LOSSLESS=0 RUN_SR=0 RUN_OFFSET=1 \
RUN_UNICORN=0 RUN_AP=0 \
./run_unicorn_lossy_geometry_curve.sh
```

For a smoke test, lower the epoch count:

```bash
EPOCHS=1 TRAIN_NUM=16 MAX_STEPS=16 TRAIN_POSQ=64.0 \
RUN_TRAIN=1 RUN_UNICORN=0 RUN_AP=0 \
./run_unicorn_lossy_geometry_curve.sh
```

The lossless training log is written to `point_pairs/unicorn_fov/logs/train_unicorn_lossless.log`. `LOSSLESS_STAGE` and `LOSSLESS_SCALE` override its default 8-stage, 5-scale training configuration. `stage=8` means that the eight octree child groups are predicted sequentially during each upsampling step. `scale=5` means that one training frame receives occupancy supervision at coordinate factors `1, 1/2, 1/4, 1/8, 1/16`; with `TRAIN_POSQ=64 mm`, these correspond to effective grids of 64, 128, 256, 512, and 1024 mm. Set `LOSSLESS_SCALE=6` if the default test point `posQuantscale=32` (2048 mm) should also be included explicitly during training. `scale` only controls training supervision and does not change checkpoint tensor shapes or limit the recursive depth of the lossless coder at test time.

## 3. Run bpp, Time, and D1/D2 PSNR

Default Unicorn test rate points are:

```text
rate_id,scale_AE,scale_SR,posQuantscale
0,0,0,1
1,0,1,2
2,0,1,4
3,0,1,8
4,0,1,16
5,0,1,32
```

The direct KITTI wrapper trains/evaluates the lossless, SR, and offset modules, so `scale_AE` stays at `0`. Override the test points with `RATES='scale_AE:scale_SR:posQuantscale,...'`.

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=0 RUN_UNICORN=1 RUN_AP=0 \
TRAIN_POSQ=64.0 \
./run_unicorn_lossy_geometry_curve.sh
```

This command loads the lossless checkpoint trained above from
`unicorn/checkpoints/lossless/epoch_last.pth`; it no longer defaults to the pretrained checkpoint under `/public/DATA/sm/Unicorn/ckpts`. To evaluate an explicitly selected checkpoint, set `LOSSLESS_LOW_CKPT=/path/to/epoch_last.pth`.

Main outputs:

```text
point_pairs/unicorn_fov/unicorn/unicorn_details.csv
point_pairs/unicorn_fov/unicorn/unicorn_average.csv
point_pairs/unicorn_fov/unicorn_rate_points.csv
point_pairs/unicorn_fov/decoded/rate_<id>/<frame>.bin
point_pairs/unicorn_fov/bitstreams/rate_<id>/<frame>.bin
```

`unicorn_average.csv` contains bpp, encoding time, decoding time, D1 PSNR, and D2 PSNR.

## 4. First-10-Frame PSNR-bpp Check

To inspect a small validation subset before running the full split, run Unicorn on the first 10 non-empty frame ids in KITTI FOV `val.txt`. The script creates `first10_split.txt`, runs all configured Unicorn rate points for those ten frames, then aggregates Baseline G-PCC, Split-GPCC, RENO, and Unicorn results.

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=4 unicorn/run_first10_psnr_bpp.sh
```

By default the selected frames are:

```text
000001 000002 000004 000005 000006 000008 000015 000019 000020 000021
```

Override the source split or rate points with environment variables:

```bash
SOURCE_SPLIT_FILE=/path/to/another_split.txt \
RATES='0:0:1,0:1:2,0:1:4,0:1:8,0:1:16,0:1:32' \
CUDA_VISIBLE_DEVICES=7 unicorn/run_first10_psnr_bpp.sh
```

Outputs:

```text
point_pairs/unicorn_first10/first10_split.txt
point_pairs/unicorn_first10/unicorn/unicorn_details.csv
point_pairs/unicorn_first10/unicorn/unicorn_average.csv
point_pairs/unicorn_first10/first10_gpcc_reno_unicorn_psnr_bpp.csv
point_pairs/unicorn_first10/first10_gpcc_reno_unicorn_d1_psnr_bpp.png
point_pairs/unicorn_first10/first10_gpcc_reno_unicorn_d2_psnr_bpp.png
```

For each method/rate, bpp is calculated as total bits divided by total input points across the ten frames; PSNR and encoding/decoding times are arithmetic means. An incomplete method/rate with fewer than ten frames is skipped. By default `UNICORN_RESUME=0`, so a rerun evaluates the currently selected checkpoints instead of reusing results from older weights; set `UNICORN_RESUME=1` only when resuming an interrupted run with the same checkpoints and rate configuration.

## 5. Evaluate AP on the KITTI Val Split

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=0 RUN_UNICORN=0 RUN_AP=1 \
./run_unicorn_lossy_geometry_curve.sh
```

Main outputs:

```text
point_pairs/unicorn_fov/logs/ap_unicorn_all_rates.log
point_pairs/unicorn_fov/unicorn_ap.csv
point_pairs/unicorn_fov/unicorn_full_curve.csv
```

## 6. One-Shot Full Run

```bash
cd /public/DATA/sm/RACO-LPCC

TRAIN_POSQ=64.0 \
RUN_TRAIN=1 RUN_UNICORN=1 RUN_AP=1 \
CUDA_VISIBLE_DEVICES=7 ./run_unicorn_lossy_geometry_curve.sh
```

If the trained Unicorn checkpoints already exist:

```bash
RUN_TRAIN=0 RUN_UNICORN=1 RUN_AP=1 \
./run_unicorn_lossy_geometry_curve.sh
```

By default this run is resumable during the Unicorn encode/decode stage:

- `UNICORN_RESUME=1` appends to `point_pairs/unicorn_fov/unicorn/unicorn_details.csv` and skips completed `frame/rate` rows on the next run.
- `GPU_GUARD=1` monitors the first id in `CUDA_VISIBLE_DEVICES` with `nvidia-smi`.
- If another compute process appears on that GPU, the current unfinished `frame/rate` is not recorded, partial averages are written from completed rows, and the script exits.
- To override the monitored physical GPU id: `GPU_GUARD_ID=7`.
- To change polling interval: `GPU_GUARD_INTERVAL=1.0`.
- To disable this behavior: `GPU_GUARD=0`.

## 7. Plot with Baseline, Split, JUQP, RENO, and Unicorn

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python unicorn/plot_unicorn_curves.py
```

Default inputs:

```text
point_pairs/baseline_fov/baseline_gpcc_curve.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
point_pairs/reno_fov/reno_full_curve.csv
point_pairs/unicorn_fov/unicorn_full_curve.csv
point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv
```

Default outputs are written to:

```text
plots_unicorn/
```

This includes AP-bpp/AP-time plots plus:

```text
plots_unicorn/d1_psnr_bpp.png
plots_unicorn/d2_psnr_bpp.png
```

## 8. Important Options

```bash
# Change Unicorn rate points: scale_AE:scale_SR:posQuantscale
RATES='0:0:1,0:1:1,0:2:1,0:3:1,0:1:2,0:2:2,0:3:2,0:2:4' \
./run_unicorn_lossy_geometry_curve.sh

# Use another output directory
OUT_DIR=/public/DATA/sm/RACO-LPCC/point_pairs/unicorn_fov \
./run_unicorn_lossy_geometry_curve.sh

# Change lossless training depth/scales or explicitly select its test checkpoint
LOSSLESS_STAGE=8 LOSSLESS_SCALE=5 \
LOSSLESS_LOW_CKPT=/path/to/lossless/epoch_last.pth \
./run_unicorn_lossy_geometry_curve.sh

# Use another detector checkpoint/config
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml \
DET_CKPT=ckpt/model_non_reflectance.pth \
./run_unicorn_lossy_geometry_curve.sh
```

## 9. Split Used for RENO

Yes. The RENO flow used KITTI FOV `train.txt` for training and KITTI FOV `val.txt` for testing/evaluation:

```text
TRAIN_SPLIT_FILE=${KITTI_ROOT}/ImageSets/train.txt
SPLIT_FILE=${KITTI_ROOT}/ImageSets/val.txt
```

The Unicorn script above uses the same split convention.
