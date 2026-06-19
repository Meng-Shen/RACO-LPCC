auto_scale_lr = dict(base_batch_size=32, enable=False)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.datasets.kitti_box_seg_dataset',
    ])
data_prefix = dict(img='', pts='', pts_instance_mask='', pts_semantic_mask='')
data_root = 'data/kitti_box_seg/'
dataset_type = 'KittiBoxSegDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='foreground',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.008
metainfo = dict(
    classes=(
        'background',
        'foreground',
    ),
    palette=[
        [
            96,
            96,
            96,
        ],
        [
            255,
            64,
            64,
        ],
    ])
model = dict(
    backbone=dict(
        base_channels=32,
        block_type='basic',
        decoder_blocks=[
            2,
            2,
            2,
            2,
        ],
        decoder_channels=[
            256,
            128,
            96,
            96,
        ],
        encoder_blocks=[
            2,
            3,
            4,
            6,
        ],
        encoder_channels=[
            32,
            64,
            128,
            256,
        ],
        in_channels=3,
        num_stages=4,
        sparseconv_backend='minkowski',
        type='MinkUNetBackbone'),
    data_preprocessor=dict(
        batch_first=False,
        max_voxels=None,
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=-1,
            max_voxels=(
                -1,
                -1,
            ),
            point_cloud_range=[
                -100,
                -100,
                -20,
                100,
                100,
                20,
            ],
            voxel_size=[
                0.05,
                0.05,
                0.05,
            ]),
        voxel_type='minkunet'),
    decode_head=dict(
        channels=96,
        dropout_ratio=0,
        ignore_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            class_weight=[
                1.0,
                8.0,
            ],
            type='mmdet.CrossEntropyLoss'),
        num_classes=2,
        type='MinkUNetHead'),
    test_cfg=dict(),
    train_cfg=dict(),
    type='MinkUNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.008, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            32,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_box_seg_infos_val.pkl',
        data_prefix=dict(
            img='', pts='', pts_instance_mask='', pts_semantic_mask=''),
        data_root=
        '/public/DATA/sm/RACO-LPCC/mmdetection3d/data/kitti_fov_box_seg/',
        ignore_index=2,
        metainfo=dict(
            classes=(
                'background',
                'foreground',
            ),
            palette=[
                [
                    96,
                    96,
                    96,
                ],
                [
                    255,
                    64,
                    64,
                ],
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                seg_3d_dtype='np.uint8',
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiBoxSegDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='SegMetric')
test_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(
        seg_3d_dtype='np.uint8',
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='kitti_box_seg_infos_train.pkl',
        data_prefix=dict(
            img='', pts='', pts_instance_mask='', pts_semantic_mask=''),
        data_root=
        '/public/DATA/sm/RACO-LPCC/mmdetection3d/data/kitti_fov_box_seg/',
        ignore_index=2,
        metainfo=dict(
            classes=(
                'background',
                'foreground',
            ),
            palette=[
                [
                    96,
                    96,
                    96,
                ],
                [
                    255,
                    64,
                    64,
                ],
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                seg_3d_dtype='np.uint8',
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                rot_range=[
                    -0.78539816,
                    0.78539816,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0.1,
                    0.1,
                    0.1,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        type='KittiBoxSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(
        seg_3d_dtype='np.uint8',
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_box_seg_infos_val.pkl',
        data_prefix=dict(
            img='', pts='', pts_instance_mask='', pts_semantic_mask=''),
        data_root=
        '/public/DATA/sm/RACO-LPCC/mmdetection3d/data/kitti_fov_box_seg/',
        ignore_index=2,
        metainfo=dict(
            classes=(
                'background',
                'foreground',
            ),
            palette=[
                [
                    96,
                    96,
                    96,
                ],
                [
                    255,
                    64,
                    64,
                ],
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                seg_3d_dtype='np.uint8',
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiBoxSegDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='SegMetric')
work_dir = '/public/DATA/sm/RACO-LPCC/mmdetection3d/work_dirs/minkunet_kitti_fov_box_seg_geometry'
