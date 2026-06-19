auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'bus',
    'person',
    'bicyclist',
    'motorcyclist',
    'road',
    'parking',
    'sidewalk',
    'other-ground',
    'building',
    'fence',
    'vegetation',
    'trunck',
    'terrian',
    'pole',
    'traffic-sign',
]
data_root = 'data/semantickitti/'
dataset_type = 'SemanticKittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
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
eval_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
labels_map = dict({
    0: 19,
    1: 19,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 19,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 19
})
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.008
metainfo = dict(
    classes=[
        'car',
        'bicycle',
        'motorcycle',
        'truck',
        'bus',
        'person',
        'bicyclist',
        'motorcyclist',
        'road',
        'parking',
        'sidewalk',
        'other-ground',
        'building',
        'fence',
        'vegetation',
        'trunck',
        'terrian',
        'pole',
        'traffic-sign',
    ],
    max_label=259,
    seg_label_mapping=dict({
        0: 19,
        1: 19,
        10: 0,
        11: 1,
        13: 4,
        15: 2,
        16: 4,
        18: 3,
        20: 4,
        252: 0,
        253: 6,
        254: 5,
        255: 7,
        256: 4,
        257: 4,
        258: 3,
        259: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        44: 9,
        48: 10,
        49: 11,
        50: 12,
        51: 13,
        52: 19,
        60: 8,
        70: 14,
        71: 15,
        72: 16,
        80: 17,
        81: 18,
        99: 19
    }))
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
        batch_first=True,
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
        ignore_index=19,
        loss_decode=dict(avg_non_ignore=True, type='mmdet.CrossEntropyLoss'),
        num_classes=19,
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
        ann_file='semantickitti_infos_val.pkl',
        backend_args=None,
        data_root='data/semantickitti/',
        ignore_index=19,
        metainfo=dict(
            classes=[
                'car',
                'bicycle',
                'motorcycle',
                'truck',
                'bus',
                'person',
                'bicyclist',
                'motorcyclist',
                'road',
                'parking',
                'sidewalk',
                'other-ground',
                'building',
                'fence',
                'vegetation',
                'trunck',
                'terrian',
                'pole',
                'traffic-sign',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
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
        type='SemanticKittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='SegMetric')
test_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
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
        ann_file='semantickitti_infos_train.pkl',
        backend_args=None,
        data_root='data/semantickitti/',
        ignore_index=19,
        metainfo=dict(
            classes=[
                'car',
                'bicycle',
                'motorcycle',
                'truck',
                'bus',
                'person',
                'bicyclist',
                'motorcyclist',
                'road',
                'parking',
                'sidewalk',
                'other-ground',
                'building',
                'fence',
                'vegetation',
                'trunck',
                'terrian',
                'pole',
                'traffic-sign',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                prob=[
                    0.5,
                    0.5,
                ],
                transforms=[
                    [
                        dict(
                            num_areas=[
                                3,
                                4,
                                5,
                                6,
                            ],
                            pitch_angles=[
                                -25,
                                3,
                            ],
                            pre_transform=[
                                dict(
                                    coord_type='LIDAR',
                                    load_dim=4,
                                    type='LoadPointsFromFile',
                                    use_dim=3),
                                dict(
                                    dataset_type='semantickitti',
                                    seg_3d_dtype='np.int32',
                                    seg_offset=65536,
                                    type='LoadAnnotations3D',
                                    with_bbox_3d=False,
                                    with_label_3d=False,
                                    with_seg_3d=True),
                                dict(type='PointSegClassMapping'),
                            ],
                            prob=1,
                            type='LaserMix'),
                    ],
                    [
                        dict(
                            instance_classes=[
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                            ],
                            pre_transform=[
                                dict(
                                    coord_type='LIDAR',
                                    load_dim=4,
                                    type='LoadPointsFromFile',
                                    use_dim=3),
                                dict(
                                    dataset_type='semantickitti',
                                    seg_3d_dtype='np.int32',
                                    seg_offset=65536,
                                    type='LoadAnnotations3D',
                                    with_bbox_3d=False,
                                    with_label_3d=False,
                                    with_seg_3d=True),
                                dict(type='PointSegClassMapping'),
                            ],
                            prob=1,
                            rotate_paste_ratio=1.0,
                            swap_ratio=0.5,
                            type='PolarMix'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                rot_range=[
                    0.0,
                    6.28318531,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        type='SemanticKittiDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        prob=[
            0.5,
            0.5,
        ],
        transforms=[
            [
                dict(
                    num_areas=[
                        3,
                        4,
                        5,
                        6,
                    ],
                    pitch_angles=[
                        -25,
                        3,
                    ],
                    pre_transform=[
                        dict(
                            coord_type='LIDAR',
                            load_dim=4,
                            type='LoadPointsFromFile',
                            use_dim=3),
                        dict(
                            dataset_type='semantickitti',
                            seg_3d_dtype='np.int32',
                            seg_offset=65536,
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True),
                        dict(type='PointSegClassMapping'),
                    ],
                    prob=1,
                    type='LaserMix'),
            ],
            [
                dict(
                    instance_classes=[
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                    ],
                    pre_transform=[
                        dict(
                            coord_type='LIDAR',
                            load_dim=4,
                            type='LoadPointsFromFile',
                            use_dim=3),
                        dict(
                            dataset_type='semantickitti',
                            seg_3d_dtype='np.int32',
                            seg_offset=65536,
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True),
                        dict(type='PointSegClassMapping'),
                    ],
                    prob=1,
                    rotate_paste_ratio=1.0,
                    swap_ratio=0.5,
                    type='PolarMix'),
            ],
        ],
        type='RandomChoice'),
    dict(
        rot_range=[
            0.0,
            6.28318531,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
tta_model = dict(type='Seg3DTTAModel')
tta_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=3),
    dict(
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        transforms=[
            [
                dict(
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=1.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=0.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=1.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
            ],
            [
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
            ],
            [
                dict(keys=[
                    'points',
                ], type='Pack3DDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='semantickitti_infos_val.pkl',
        backend_args=None,
        data_root='data/semantickitti/',
        ignore_index=19,
        metainfo=dict(
            classes=[
                'car',
                'bicycle',
                'motorcycle',
                'truck',
                'bus',
                'person',
                'bicyclist',
                'motorcyclist',
                'road',
                'parking',
                'sidewalk',
                'other-ground',
                'building',
                'fence',
                'vegetation',
                'trunck',
                'terrian',
                'pole',
                'traffic-sign',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=3),
            dict(
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
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
        type='SemanticKittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='SegMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/minkunet34_w32_minkowski_geometry_8xb2-laser-polar-mix-3x_semantickitti'
