

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

  python train_cost_proxy.py \
  --velodyne_dir ../data/kitti/training/velodyne \
  --train_split ../data/kitti/ImageSets/train.txt \
  --ap_csv split_AP_train.csv \
  --test_split ../data/kitti/ImageSets/val.txt \
  --test_ap_csv test/split_AP.csv \
  --thresholds "0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075" \
  --test_every 5 \
  --out_dir router_work_dirs/cost_proxy1 \
  --epochs 120 \
  --batch_size 8 \
  --workers 4 \
  --voxel_size 0.16 0.16 0.16 \
  --point_cloud_range 0 -40 -3 70.4 40 1 \
  --max_voxels 50000 \
  --feat_dim 256 \
  --ap_drop_scale 100 \
  --lambda_threshold 0.0 \
  --ap_weights 10.0 1.0 1.0 \
  --allow_negative_cost \
  --lr 5e-4 \
  --jitter_std 0.005 \
  --weight_decay 5e-4 \
  --val_split router_val.txt \
  --val_ap_csv split_AP_train.csv