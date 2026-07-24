# verify_juqp.sh 使用说明

`verify_juqp.sh` 用现有脚本验证 GPCC baseline 与 oracle/JUQP 理想曲线：

1. 用 GPCC baseline 的全局量化精度，在 KITTI FOV val split 上跑目标检测。这里调用 `OpenPCDet/tools/test_pos.py`，只做直接量化/反量化，不调用真实 G-PCC 压缩。
2. 用得到的各量化精度 `result.pkl` 调用 `OpenPCDet/tools/new_split.py`，计算每帧在各量化精度下替换预测结果后的 3 个 AP：`Car_3d_AP_R40_moderate`、`Pedestrian_3d_AP_R40_moderate`、`Cyclist_3d_AP_R40_moderate`。
3. 用最大码率作为 Label 0，计算各帧各量化精度相对最大码率的 AP drop，再调用 `compute_oracle_router_curve.py` 按 `AP_drop + lambda * bpp` 的 oracle/JUQP 策略为每帧每个 lambda 选标签。
4. 输出 GPCC baseline AP-bpp 曲线和 oracle/JUQP 理想 AP-bpp 曲线图。

默认量化精度优先从已有 CSV 读取：

```text
point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv
```

脚本也默认复用同目录下已有的 GPCC bpp/time 统计：

```text
point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv
point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv
```

脚本还默认复用之前跑基础方法 AP-bpp 时留下的检测结果和 AP CSV：

```text
point_pairs/baseline_fov/baseline_ap.csv
OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/default/eval/epoch_no_number/val/default/scale_*/result.pkl
```

当前已检查到 7 个 `scale_* / result.pkl`，每个都是 3769 帧，范围为 `000001` 到 `007480`，可以直接用于后续逐帧 AP sensitivity 和 oracle/JUQP 计算。

因此 `RUN_AP` 和 `RUN_GPCC` 默认都是 `0`，不会重复跑检测或真实 G-PCC bpp 统计。读取量化精度时优先读 `posQuantscale`，没有该列时读 `scale`。如果该 CSV 不存在，则使用：

```text
1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512
```

当前仓库中这个 CSV 对应的实际小数值是：

```text
0.015625,0.01171875,0.0078125,0.005859375,0.00390625,0.0029296875,0.001953125
```

## 一键运行

```bash
cd /public/DATA/sm/RACO-LPCC

CUDA_VISIBLE_DEVICES=6 ./verify_juqp.sh
```

## 修改量化精度

`SCALES` 可以直接覆盖默认量化精度，逗号分隔，支持小数或分数：

```bash
CUDA_VISIBLE_DEVICES=4 \
SCALES='1/64,1/128,1/256,1/512' \
./verify_juqp.sh
```

如果要从另一个已有 GPCC CSV 读取：

```bash
EXISTING_GPCC_CSV=point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv \
./verify_juqp.sh
```

脚本会把全局精度转成 oracle/JUQP 使用的 `QUANT_MAP`，即每个 label 都是 `scale,scale`。也可以手动覆盖：

```bash
QUANT_MAP='0.015625,0.015625;0.0078125,0.0078125;0.00390625,0.00390625' \
SCALES='0.015625,0.0078125,0.00390625' \
./verify_juqp.sh
```

## 分步跳过

已存在中间结果时可以跳过耗时步骤：

```bash
RUN_AP=0 RUN_GPCC=0 ./verify_juqp.sh
```

可用开关：

```text
RUN_AP=0
RUN_GPCC=0
RUN_SENSITIVITY=1
RUN_ORACLE=1
RUN_PLOT=1
```

即默认一键运行会直接复用已有基础方法结果，只从 `new_split.py` 的逐帧 AP sensitivity 开始往后跑。脚本会在自己的输出目录下生成 `combo_*` 软链接，供 `new_split.py` 和 `compute_oracle_router_curve.py` 使用，不会覆盖原始 `OpenPCDet/output` 里的结果：

```text
point_pairs/verify_juqp_fov/gpcc_scale_eval_links/
```

如果只改 `LAGRANGE_LAMBDAS` 或 `OBJECTIVE`，并且检测结果、GPCC 明细和 `val_ap_sensitivity.csv` 已经存在，只需要重跑 oracle 和画图：

```bash
RUN_AP=0 \
RUN_GPCC=0 \
RUN_SENSITIVITY=0 \
RUN_ORACLE=1 \
RUN_PLOT=1 \
LAGRANGE_LAMBDAS='0,0.0001,0.00025,0.0005,0.001,0.002' \
./verify_juqp.sh
```

如果 oracle 结果也已经存在，只想重新画图：

```bash
RUN_AP=0 RUN_GPCC=0 RUN_SENSITIVITY=0 RUN_ORACLE=0 RUN_PLOT=1 ./verify_juqp.sh
```

常用参数：

```text
CFG_FILE=cfgs/kitti_models/pv_rcnn_fov_geometry.yaml
DET_CKPT=ckpt/model_non_reflectance.pth
BATCH_SIZE=8
WORKERS=4
OBJECTIVE=Car
LAGRANGE_LAMBDAS=0,0.00025,0.0005,0.001,0.002,0.004,0.008,0.016,0.032
BASELINE_AP_CSV=point_pairs/baseline_fov/baseline_ap.csv
PKL_EVAL_DIR=point_pairs/verify_juqp_fov/gpcc_scale_eval_links
OUT_DIR=point_pairs/verify_juqp_fov
```

## 输出

默认输出目录：

```text
point_pairs/verify_juqp_fov/
```

主要文件：

```text
baseline_ap.csv
baseline_gpcc_curve.csv
val_ap_sensitivity.csv
oracle_juqp/oracle_router_curve.csv
oracle_juqp/oracle_average_results.csv
oracle_juqp/oracle_all_details.csv
oracle_juqp/oracle_rate_*_labels.csv
oracle_juqp/mixed_result_pkls/oracle_rate_*_result.pkl
plots/ap_bpp_car.png
plots/ap_bpp_pedestrian.png
plots/ap_bpp_cyclist.png
```

默认 GPCC bpp/time 输入来自：

```text
point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv
point_pairs/baseline_fov/gpcc/gpcc_baseline_average.csv
```

默认 baseline AP 和 `result.pkl` 输入来自：

```text
point_pairs/baseline_fov/baseline_ap.csv
OpenPCDet/output/kitti_models/pv_rcnn_fov_geometry/default/eval/epoch_no_number/val/default/
```

`oracle_rate_*_labels.csv` 中的 `jucp_label` 就是对应 lambda 下每帧选择的 JUQP 标签。标签 0 是最大码率，后续标签按 `SCALES` 顺序对应更低码率。
