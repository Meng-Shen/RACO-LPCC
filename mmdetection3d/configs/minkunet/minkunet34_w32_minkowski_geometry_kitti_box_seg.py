_base_ = [
    '../_base_/models/minkunet.py', '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet3d.datasets.kitti_box_seg_dataset'],
    allow_failed_imports=False)

data_root = 'data/kitti_box_seg/'
dataset_type = 'KittiBoxSegDataset'
metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[96, 96, 96], [255, 64, 64]])

model = dict(
    data_preprocessor=dict(max_voxels=None),
    backbone=dict(
        in_channels=3,
        encoder_blocks=[2, 3, 4, 6],
        sparseconv_backend='minkowski'),
    decode_head=dict(
        num_classes=2,
        ignore_index=2,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            class_weight=[1.0, 20.0],
            avg_non_ignore=True)))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=3),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=3),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8'),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

data_prefix = dict(
    pts='', pts_semantic_mask='', img='', pts_instance_mask='')

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_box_seg_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=2))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_box_seg_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=2,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='foreground',
        rule='greater',
        max_keep_ckpts=3))

work_dir = './work_dirs/minkunet_kitti_box_seg_geometry'
