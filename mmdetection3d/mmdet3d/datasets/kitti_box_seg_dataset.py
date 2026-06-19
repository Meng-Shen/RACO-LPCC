"""KITTI box-supervised foreground/background segmentation dataset."""

from mmdet3d.registry import DATASETS

from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class KittiBoxSegDataset(Seg3DDataset):
    """Two-class point segmentation labels generated from KITTI 3D boxes."""

    METAINFO = {
        'classes': ('background', 'foreground'),
        'palette': [[96, 96, 96], [255, 64, 64]],
        'seg_valid_class_ids': (0, 1),
        'seg_all_class_ids': (0, 1, 2),
    }
