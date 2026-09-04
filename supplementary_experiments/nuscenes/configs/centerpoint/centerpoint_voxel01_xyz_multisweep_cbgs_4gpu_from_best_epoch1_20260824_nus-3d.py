_base_ = [
    './centerpoint_voxel01_xyz_multisweep_cbgs_from_epoch2_20260824_nus-3d.py'
]

# Continue the full multi-sweep, geometry-only experiment from the best
# checkpoint produced by its first single-GPU epoch.  The optimizer is reset
# because this run changes the global batch from 4 to 16 across four GPUs.
work_dir = (
    '/home/sm/raco_rate_aware_nuscenes_20260822/experiments/'
    'centerpoint_voxel01_xyz_multisweep_cbgs_4gpu_from_best_epoch1_5e_20260824'
)
pretrained = (
    '/home/sm/raco_rate_aware_nuscenes_20260822/experiments/'
    'centerpoint_voxel01_xyz_multisweep_cbgs_from_epoch2_6e_20260824/'
    'best_NuScenes metric_pred_instances_3d_NuScenes_mAP_epoch_1.pth'
)
load_from = pretrained
resume = False

# Keep batch_size=4 per GPU (global batch 16).  Linear LR scaling preserves the
# per-sample update magnitude of the preceding one-GPU batch-4 fine-tuning.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=4e-5))

# One full epoch is already represented by the initialization checkpoint; run
# the remaining five epochs and validate/save by official overall nuScenes mAP.
train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=5,
        by_epoch=True,
        milestones=[3, 4],
        gamma=0.1)
]
default_hooks = dict(
    logger=dict(interval=25),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater'))
auto_scale_lr = dict(enable=False, base_batch_size=16)
