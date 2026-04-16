安装参考：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d)



点云分割命令：
cd mmdetection3d
python demo/pcd_seg_demo.py \
    demo/data/000000.bin \
    configs/minkunet/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py \
    ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth \
    --no-save-vis \
    --no-save-pred
可视化分割结果：
python demo/visualize.py demo/data/000000.bin output/preds/000000.json

track环境安装注意事项：
torch一定要根据pytorch官网指令安装指定版本：1.10.0 cuda11.1适配rtx3090硬件
代码修改：
将models.MBPTrack.transformer.py中的_LinearWithBias改成torch.nn.linear
目标追踪测试命令：
cd MBPTrack3D
python main.py \
    configs/mbptrack_kitti_car_cfg.yaml \
    --phase test \
    --resume_from pretrained/mbptrack_kitti_car.ckpt
额外工作：修改yaml文件中的数据集路径、可在datasets/kitti_mem.py中修改测试集定义

标注框可视化：
python visualize_kitti.py \
    training/velodyne/0012/000000.bin \
    training/label_02/0012.txt \
    training/calib/0012.txt

点云压缩：
cd Unicorn/lossy_geometry
python test.py \
    --testdata='../../../Dataset/kitti/training/velodyne/0012' \
    --outdir='output' \
    --testdata_num=100 \
    --testdata_seqs='random' \
    --max_num=120000 \
    --start_index=0 \
    --interval=1 \
    --resolution=80000 \
    --threshold=1.15 \
    --threshold_lossy=0 \
    --kernel_size=5 \
    --block_type='tf' \
    --ckptdir_low='../ckpts/geometry/kitti/kitti2cm/conv/epoch_last.pth' \
    --ckptdir_sr_low='' \
    --ckptdir_ae_low='' \
    --ckptdir_high='../ckpts/geometry/kitti/kitti1mm/tf/epoch_last.pth' \
    --ckptdir_sr_high='' \
    --ckptdir_ae_high='' \
    --ckptdir_offset='../ckpts/geometry/kitti/kitti1mm/offset/epoch_last.pth' \
    --bitrate_mode=2 \
    --prefix='kitti1mm'  

gpcc压缩点云：
cd GPCC
python test.py \
    --testdata='../mmdetection3d/demo/data' \
    --resolution=80000 \
    --output='output' \
    --results='results'

python test_attr.py \
    --testdata='../mmdetection3d/demo/data' \
    --resolution=80000 \
    --output='output' \
    --results='results'

python encode_attr.py \
    --testdata='../../Dataset/kitti/training/velodyne/0020' \
    --bitstream='bitstream' \
    --result='encode_results'

python decode_attr.py --bitstream ./bitstream --output ./output --result ./decode_results

python psnr_attr.py --origin ../mmdetection3d/demo/data --recon ./output/R0 --result psnr_results

python print_pc.py \
    --testdata='../mmdetection3d/demo/data'

python single_split.py \
    ../mmdetection3d/demo/data/000000.bin \
    ../mmdetection3d/configs/minkunet/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py \
    ../mmdetection3d/ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth \
    --fg-quant 0.125 \
    --bg-quant 0.125 \
    --gpcc-cfg kitti.cfg \
    --out-dir output

python test_split.py \
    --testdata='../../Dataset/kitti/training/velodyne/0012' \
    --model='../mmdetection3d/configs/minkunet/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py' \
    --weights='../mmdetection3d/ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth'


分配反射率：
cd RACO-LPCC
python nn.py \
    --input ./GPCC/output \
    --ref ../Dataset/kitti/training/velodyne/0012 \
    --output ../Dataset/kitti/training/w_0012 \
    --mode r6

测试点云关系
python check.py \
    --dir1 mmdetection3d/demo/data/000000.bin \
    --dir2 GPCC/output/000000_R0.bin

!!! 可以只考虑点云量化步长、不进行任何方法的压缩，可以考虑引入
坐标偏移模块。
！！！只把邻域点云按顺序排列，在MBPTrack3D/tasks/mbp_task.py225行已改好。

cd tools
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 --ckpt ckpt/latest_model.pth
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 --ckpt /public/DATA/sm/RACO-LPCC/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/checkpoint_epoch_10.pth
python single_frame_eval.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt ckpt/pv_rcnn_8369.pth \
    --frame_id 000008 \
    --iou_thresh 0.5

python jucp.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt ckpt/latest_model.pth \
    --split_file ../data/kitti/ImageSets/val.txt \
    --iou_thresh 0.5 \
    --out_csv jucp_labels.csv

python test_jucp.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --batch_size 8 \
    --ckpt ckpt/latest_model.pth \
    --jucp_csv jucp_labels.csv

python count.py 

python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --pretrained_model ckpt/pv_rcnn_8369.pth

python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

目标检测环境安装问题：
旧setuptools 82.0.1 ->降低
pip install spconv-cu116
pip install kornia==0.6.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

面向目标检测的点云压缩：
gpcc原方法多码率压缩：
python test_gpcc.py \
    --testdata='../OpenPCDet/data/kitti/training/velodyne'

python quantize.py \
    --testdata='../OpenPCDet/data/kitti/training/velodyne'