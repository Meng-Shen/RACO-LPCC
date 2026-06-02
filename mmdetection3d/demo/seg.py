import numpy as np
from mmdet3d.apis import LidarSeg3DInferencer

def parse_args():
    call_args = {
        "pcd": "../mmdetection3d/demo/data/000000.bin",          
        "model": "../mmdetection3d/configs/minkunet/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py",
        "weights": "../mmdetection3d/ckpt/minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth",    
        "device": "cuda:0",  
        "out_dir": "output",
        "show": False,       
        "wait_time": -1.0,   
        "no_save_vis": False,
        "no_save_pred": False,
        "print_result": False 
    }

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def split(init_args, call_args):
    inferencer = LidarSeg3DInferencer(**init_args)
    result = inferencer(**call_args)
    
    points = result['visualization'][0]
    pred_labels = np.array(result['predictions'][0]['pts_semantic_mask'])

    foreground_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18] 
    fg_mask = np.isin(pred_labels, foreground_indices)
    bg_mask = ~fg_mask
    foreground_coords = points[fg_mask]
    background_coords = points[bg_mask]