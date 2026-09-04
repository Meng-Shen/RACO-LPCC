_base_ = [
    '/home/sm/raco_rate_aware_nuscenes_20260822/configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_nus-3d.py'
]

data_root = '/home/sm/raco_rate_aware_nuscenes_20260822/data/nuscenes/'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'car',
    'truck', 'trailer', 'bus', 'construction_vehicle'
]
metainfo = dict(classes=class_names)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

xyz_keyframe = dict(
    type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
    backend_args=None)
xyz_sweeps_train = dict(
    type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5,
    use_dim=[0, 1, 2], pad_empty_sweeps=True, remove_close=True,
    backend_args=None)
xyz_sweeps_test = dict(**xyz_sweeps_train, test_mode=True)

train_pipeline = [
    xyz_keyframe,
    xyz_sweeps_train,
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D', sync_2d=False,
        flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    xyz_keyframe,
    xyz_sweeps_test,
    dict(
        type='MultiScaleFlipAug3D', img_scale=(1333, 800),
        pts_scale_ratio=1, flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans', rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]
eval_pipeline = [xyz_keyframe, xyz_sweeps_test,
                 dict(type='Pack3DDetInputs', keys=['points'])]

model = dict(pts_voxel_encoder=dict(in_channels=3))

train_dataloader = dict(
    batch_size=2, num_workers=3, persistent_workers=True,
    dataset=dict(
        data_root=data_root, pipeline=train_pipeline, metainfo=metainfo,
        data_prefix=data_prefix))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(
        data_root=data_root, pipeline=test_pipeline, metainfo=metainfo,
        data_prefix=data_prefix))
test_dataloader = val_dataloader

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False,
        begin=0, end=500),
    dict(
        type='CosineAnnealingLR', begin=0, end=8, by_epoch=True,
        T_max=8, eta_min=1e-6),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=8,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=25))
auto_scale_lr = dict(enable=False, base_batch_size=32)
load_from = (
    '/home/sm/raco_rate_aware_nuscenes_20260822/checkpoints/'
    'transfer_detectors_20260825/ssn_xyz_adapted.pth')
