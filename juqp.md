cd OpenPCDet
cd tools

# 得到逐点云逐码率预测框文件，搭配new_split.py使用
python test_split.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt ckpt/latest_model.pth \
    --batch_size 8

# 根据各文件各压缩码率的预测框中间结果求每帧每个码率下的整体AP值（其他帧取最大码率）
python new_split.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --split_file ../data/kitti/ImageSets/val.txt \
    --eval_dir ../output/kitti_models/pv_rcnn/default/eval/epoch_no_number/val/default \
    --out_csv split_AP.csv \
    --workers 64

# 根据每帧每码率AP值求jucp，需自己定近无损阈值
python jucp_split.py --ap_csv split_AP.csv --out_csv jucp0.0045_0.05_0.075.csv

# 根据jucp标签跑语义分割协助压缩方案，得到AP性能
python test_jucp_split.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --batch_size 8 \
    --ckpt ckpt/latest_model.pth \
    --jucp_csv jucp0.0045_0.05_0.075.csv \
    --mask_dir ../output/eval/seg_masks \
    --workers 4