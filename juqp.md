

QUANT_MAP='' ./juqp.sh

RUN_JUCP_SPLIT=0 RUN_TEST_JUCP_SPLIT=0 ./juqp.sh
RUN_JUCP_SPLIT=0 RUN_TEST_JUCP_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0 \
JUCP_PED_THRESHOLD=0 \
JUCP_CYC_THRESHOLD=0 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0.001 \
JUCP_PED_THRESHOLD=0.01 \
JUCP_CYC_THRESHOLD=0.02 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0.0015 \
JUCP_PED_THRESHOLD=0.02 \
JUCP_CYC_THRESHOLD=0.035 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0.0025 \
JUCP_PED_THRESHOLD=0.03 \
JUCP_CYC_THRESHOLD=0.045 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0.0035 \
JUCP_PED_THRESHOLD=0.04 \
JUCP_CYC_THRESHOLD=0.06 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

JUCP_CAR_THRESHOLD=0.0045 \
JUCP_PED_THRESHOLD=0.05 \
JUCP_CYC_THRESHOLD=0.075 \
RUN_TEST_SPLIT=0 RUN_NEW_SPLIT=0 ./juqp_train.sh

# 生成前背景掩码
cd OpenPCDet/tools

python generate_masks.py \
    --seg_cfg_file ../../mmdetection3d/configs/minkunet/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py \
    --seg_ckpt ../../mmdetection3d/ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth


# 训练路由代理
cd /public/DATA/sm/RACO-LPCC/OpenPCDet/tools

# CUDA_VISIBLE_DEVICES 指定使用哪一张 GPU。
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python train_cost_proxy.py \
  --velodyne_dir ../data/kitti/training/velodyne \
  --train_split ../data/kitti/ImageSets/train.txt \
  --ap_csv split_AP_train.csv \
  --test_split ../data/kitti/ImageSets/val.txt \
  --test_ap_csv test/split_AP.csv \
  --split_test_for_val \
  --test_val_ratio 0.5 \
  --thresholds "0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075" \
  --test_every 0 \
  --out_dir router_work_dirs/cost_proxy_res_mono \
  --epochs 120 \
  --batch_size 8 \
  --workers 4 \
  --voxel_size 0.16 0.16 0.16 \
  --point_cloud_range 0 -40 -3 70.4 40 1 \
  --max_voxels 50000 \
  --feat_dim 256 \
  --ap_drop_scale 100 \
  --lambda_threshold 0.1 \
  --ap_weights 10.0 1.0 1.0 \
  --lr 5e-4 \
  --jitter_std 0.005 \
  --weight_decay 5e-4 \
  --calibrate_cost \
  --calibration_epochs 20 \
  --calibration_lr 1e-2

# 只做校准和最终测试，不重新训练主网络。
# 注意：当前 best.pth 已经被后一次训练覆盖；这里使用接近原最佳点的 epoch_030.pth。
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python train_cost_proxy.py \
  --velodyne_dir ../data/kitti/training/velodyne \
  --train_split ../data/kitti/ImageSets/train.txt \
  --ap_csv split_AP_train.csv \
  --test_split ../data/kitti/ImageSets/val.txt \
  --test_ap_csv test/split_AP.csv \
  --split_test_for_val \
  --test_val_ratio 0.5 \
  --thresholds "0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075" \
  --test_every 0 \
  --out_dir router_work_dirs/cost_proxy_res_mono \
  --epochs 0 \
  --batch_size 8 \
  --workers 4 \
  --voxel_size 0.16 0.16 0.16 \
  --point_cloud_range 0 -40 -3 70.4 40 1 \
  --max_voxels 50000 \
  --feat_dim 256 \
  --ap_drop_scale 100 \
  --lambda_threshold 0.1 \
  --ap_weights 10.0 1.0 1.0 \
  --lr 5e-4 \
  --jitter_std 0.005 \
  --weight_decay 5e-4 \
  --calibration_only \
  --calibrate_cost \
  --calibration_ckpt epoch_030.pth \
  --calibration_epochs 20 \
  --calibration_lr 1e-2
