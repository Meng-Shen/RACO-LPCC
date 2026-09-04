_base_ = [
    '/home/sm/sunrgbd_lite_s3_20260828/mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py'
]

data_root = '/home/sm/sunrgbd_lite_s3_20260828/data/sunrgbd/'
work_dir = '/home/sm/sunrgbd_lite_s3_20260828/experiments/votenet_geometry_finetune'

# The upstream SUN RGB-D VoteNet recipe is already geometry-only: the loader
# reads XYZ and derives only the height feature.  Keep RGB outside the model.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(data_root=data_root),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(data_root=data_root),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(data_root=data_root),
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=12,
        by_epoch=True,
        T_max=12,
        eta_min=1e-5,
    )
]

load_from = '/home/sm/sunrgbd_lite_s3_20260828/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=4,
        save_best='mAP_0.25',
        rule='greater',
        save_last=True,
    ),
)
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mAP_0.25',
        rule='greater',
        min_delta=0.001,
        patience=4,
        strict=False,
    )
]

randomness = dict(seed=20260828, deterministic=True)
