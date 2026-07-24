# RENO replication

This replication follows the existing G-PCC baseline style in `RACO-LPCC`: the driver script lives in this repo, reads KITTI `.bin` directly, imports the RENO model code from `/public/DATA/sm/RENO`, and writes bpp/time/AP/PSNR CSV outputs. It does not call RENO upstream `compress.py`, `decompress.py`, or `eval.py`.

Two implementation details matter:

- The rate points use the same 8 quantization scales as the sparse SR module: `1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512,1/2048`.
- Coordinate preprocessing is changed from upstream RENO's fixed `+131072` shift to the same per-frame offset logic used by the G-PCC baseline.
- RENO is lossless after its initial coordinate quantization. Therefore AP evaluation does not need saved decoded files. The AP script directly applies the same quantization in the OpenPCDet loader, like the G-PCC `test_pos.py` flow.

## Scripts

```text
RACO-LPCC/run_reno_baseline_curve.sh
RACO-LPCC/reno/train_kitti.py
RACO-LPCC/reno/reno_rates.py
RACO-LPCC/reno/test_reno_pos.py
RACO-LPCC/reno/parse_reno_ap_logs.py
RACO-LPCC/reno/merge_reno_curve.py
RACO-LPCC/reno/plot_reno_curves.py
```

`reno_rates.py` performs the actual RENO encode/decode loop with `Network`, `FOG/FCG`, and `torchac`. It reads KITTI `.bin`, writes RENO bitstreams, measures bpp/time, computes D1/D2 PSNR, and removes temporary PSNR PLY files immediately.

`test_reno_pos.py` evaluates AP without reading decoded files. For each scale it computes:

```text
posQ = 1 / scale
coords_mm = round(xyz * 1000)
offset = min(coords_mm, axis=0)
coords_scaled = coords_mm - offset
q = round(coords_scaled / posQ)
xyz_dec = (unique(q) * posQ + offset) / 1000
```

This matches the existing GPCC baseline's coordinate normalization style while still using RENO's entropy-coded occupancy model. The occupancy coding is lossless for those quantized coordinates.

## 0. Environment Variables

```bash
cd /public/DATA/sm/RACO-LPCC

export RENO_ROOT=/public/DATA/sm/RENO
export PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python
export RENO_PYTHON_BIN=/home/sm/miniconda3/envs/SparsePCGC/bin/python

export KITTI_ROOT=/public/DATA/sm/RACO-LPCC/OpenPCDet/data/kitti_fov
export KITTI_VELODYNE=${KITTI_ROOT}/training/velodyne
export SPLIT_FILE=${KITTI_ROOT}/ImageSets/val.txt
export TRAIN_SPLIT_FILE=${KITTI_ROOT}/ImageSets/train.txt

export SCALES='1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512,1/2048'
```

## 1. Prepare FOV KITTI

```bash
cd /public/DATA/sm/RACO-LPCC
./prepare_kitti_fov.sh
```

## 2. Train RENO on KITTI

RENO has no released KITTI checkpoint requirement for this run, so train one checkpoint from KITTI training frames first. This uses `RACO-LPCC/reno/train_kitti.py`, not RENO upstream `train.py`.

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=1 RUN_RENO=0 RUN_AP=0 MAX_STEPS=170000 TRAIN_POSQ=64.0 ./run_reno_baseline_curve.sh
```

Output checkpoint:

```text
point_pairs/reno_fov/model/ckpt.pt
```

For a smoke test only, lower `MAX_STEPS`, for example `MAX_STEPS=1000`.

## 3. Run RENO Rate, Time, and D1/D2 PSNR

This step mirrors `GPCC/baseline_rates.py`, but uses RENO. It runs all 8 scales on the validation split, measures aggregate bpp as `sum(bits) / sum(points)`, measures average encoding/decoding time, and computes D1/D2 PSNR. Temporary files for `pc_error_d` are deleted per frame.

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=0 RUN_RENO=1 RUN_AP=0 ./run_reno_baseline_curve.sh
```

Main outputs:

```text
point_pairs/reno_fov/reno/reno_details.csv
point_pairs/reno_fov/reno/reno_average.csv
point_pairs/reno_fov/bitstreams/rate_<id>/<frame>.bin
```

`reno_average.csv` contains:

```text
rate_id,posQ,num_frames,total_points,total_bits,scale,scale_label,bpp,enc_time,dec_time,d1_psnr,d2_psnr
```

## 4. Evaluate Detection AP Without Decoded Files

This step evaluates PV-RCNN once over all RENO rate points. It monkey-patches OpenPCDet's KITTI `get_lidar` method and directly applies RENO-equivalent quantization in memory.

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=0 RUN_RENO=0 RUN_AP=1 ./run_reno_baseline_curve.sh
```

Main outputs:

```text
point_pairs/reno_fov/logs/ap_reno_all_rates.log
point_pairs/reno_fov/reno_ap.csv
point_pairs/reno_fov/reno_full_curve.csv
```

`reno_ap.csv` contains KITTI R40 AP for Car, Pedestrian, and Cyclist. `reno_full_curve.csv` merges AP with bpp/time/D1/D2.

## 5. Quick Single-Frame PSNR-bpp Check

To quickly inspect one frame before running the whole validation split, run RENO on one KITTI point cloud and combine it with existing per-frame Baseline G-PCC and Split-GPCC PSNR/bpp details:

```bash
cd /public/DATA/sm/RACO-LPCC

reno/run_single_frame_psnr_bpp.sh point_pairs/reno_fov/model/ckpt.pt 000001
```

The second argument can also be a `.bin` path:

```bash
reno/run_single_frame_psnr_bpp.sh /path/to/ckpt.pt OpenPCDet/data/kitti_fov/training/velodyne/000001.bin
```

Default outputs:

```text
point_pairs/reno_single_frame/<frame_id>/<frame_id>_single_frame_psnr_bpp.csv
point_pairs/reno_single_frame/<frame_id>/<frame_id>_d1_psnr_bpp.png
point_pairs/reno_single_frame/<frame_id>/reno/reno_details.csv
```

This script computes RENO for the selected frame only and plots comparable D1 PSNR-bpp. For Baseline G-PCC and Split-GPCC, it reads existing single-frame bpp rows from `gpcc_baseline_details.csv` and `split_all_details.csv`, and existing single-frame PSNR rows from `point_pairs/psnr_bpp/*_psnr_details.csv`.

## 6. Plot AP Curves with RENO

After `reno_full_curve.csv` is generated, draw AP-bpp, AP-encoding-time, AP-decoding-time, and PSNR-bpp curves with Baseline G-PCC, Split-GPCC, JUQP Router, and RENO together:

```bash
cd /public/DATA/sm/RACO-LPCC

/home/sm/miniconda3/envs/SparsePCGC/bin/python reno/plot_reno_curves.py
```

Default inputs:

```text
point_pairs/baseline_fov/baseline_gpcc_curve.csv
point_pairs/split_gpcc_fov/split_gpcc_curve.csv
point_pairs/router_gpcc_fov/router_gpcc_curve.csv
point_pairs/reno_fov/reno_full_curve.csv
point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv
```

Default outputs are written to `plots_reno/`:

```text
plots_reno/ap_bpp_car.png
plots_reno/ap_bpp_pedestrian.png
plots_reno/ap_bpp_cyclist.png
plots_reno/ap_enctime_car.png
plots_reno/ap_enctime_pedestrian.png
plots_reno/ap_enctime_cyclist.png
plots_reno/ap_dectime_car.png
plots_reno/ap_dectime_pedestrian.png
plots_reno/ap_dectime_cyclist.png
plots_reno/d1_psnr_bpp.png
plots_reno/d2_psnr_bpp.png
```

To override input or output paths:

```bash
/home/sm/miniconda3/envs/SparsePCGC/bin/python reno/plot_reno_curves.py \
  --baseline_csv point_pairs/baseline_fov/baseline_gpcc_curve.csv \
  --split_csv point_pairs/split_gpcc_fov/split_gpcc_curve.csv \
  --juqp_csv point_pairs/router_gpcc_fov/router_gpcc_curve.csv \
  --reno_csv point_pairs/reno_fov/reno_full_curve.csv \
  --psnr_csv point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv \
  --out_dir plots_reno
```

## 7. One-Shot Full Run

After environment setup, this runs training, RENO codec evaluation, AP evaluation, and merge in one command.

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=1 RUN_RENO=1 RUN_AP=1 MAX_STEPS=170000 ./run_reno_baseline_curve.sh
```

If a trained checkpoint already exists, skip training:

```bash
cd /public/DATA/sm/RACO-LPCC

RUN_TRAIN=0 RUN_RENO=1 RUN_AP=1 ./run_reno_baseline_curve.sh
```

## 8. Important Options

```bash
# Use another RENO checkpoint for rate/time evaluation
RUN_TRAIN=0 RUN_RENO=1 RUN_AP=1 ./run_reno_baseline_curve.sh --reno_ckpt /path/to/ckpt.pt

# Equivalent environment-variable form
RENO_CKPT=/path/to/ckpt.pt RUN_TRAIN=0 RUN_RENO=1 RUN_AP=1 ./run_reno_baseline_curve.sh

# Change output directory
OUT_DIR=/public/DATA/sm/RACO-LPCC/point_pairs/reno_fov ./run_reno_baseline_curve.sh

# Change rate points, using GPCC/SR-style scales
SCALES='1/64,1/128,1/256,1/512' ./run_reno_baseline_curve.sh

# Change detector checkpoint/config
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml DET_CKPT=ckpt/model_non_reflectance.pth ./run_reno_baseline_curve.sh

# Run only RENO bpp/time without PSNR for debugging
${RENO_PYTHON_BIN} reno/reno_rates.py   --reno_root ${RENO_ROOT}   --testdata ${KITTI_VELODYNE}   --split_file ${SPLIT_FILE}   --scales ${SCALES}   --ckpt point_pairs/reno_fov/model/ckpt.pt   --results point_pairs/reno_fov/reno   --tmp_dir point_pairs/reno_fov/tmp   --bitstream_dir point_pairs/reno_fov/bitstreams   --kitti_root ${KITTI_ROOT}   --no_psnr
```

## 9. Final Files

```text
point_pairs/reno_fov/reno/reno_average.csv       # bpp, enc/dec time, D1/D2 PSNR
point_pairs/reno_fov/reno/reno_details.csv       # per-frame details
point_pairs/reno_fov/reno_ap.csv                 # AP per scale
point_pairs/reno_fov/reno_full_curve.csv         # merged curve table
point_pairs/reno_fov/bitstreams/rate_<id>/       # RENO bitstreams
plots_reno/ap_*_*.png                                # AP curves with RENO
point_pairs/reno_single_frame/<frame_id>/             # quick single-frame PSNR-bpp check
```
