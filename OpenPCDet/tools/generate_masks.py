import os
import argparse
import numpy as np
from tqdm import tqdm
from mmdet3d.apis import LidarSeg3DInferencer

def main():
    parser = argparse.ArgumentParser(description="Offline Semantic Mask Generation")
    parser.add_argument('--val_txt', type=str, default='../data/kitti/ImageSets/val.txt', help='Path to val.txt')
    parser.add_argument('--bin_dir', type=str, default='../data/kitti/training/velodyne', help='Path to velodyne bins')
    parser.add_argument('--out_dir', type=str, default='../output/eval/seg_masks', help='Output directory for .npy masks')
    
    # 必须传入 mmdet3d 的配置和权重
    parser.add_argument('--seg_cfg_file', type=str, required=True)
    parser.add_argument('--seg_ckpt', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.val_txt, 'r') as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[*] Loading Semantic Model from {args.seg_ckpt}...")
    # 强制单卡推理，环境极其干净
    inferencer = LidarSeg3DInferencer(model=args.seg_cfg_file, weights=args.seg_ckpt, device='cuda:0')

    for vid in tqdm(val_ids, desc="Generating Masks"):
        out_path = os.path.join(args.out_dir, f"{vid}.npy")
        if os.path.exists(out_path):
            continue
            
        bin_path = os.path.join(args.bin_dir, f"{vid}.bin")
        
        # 纯净的单帧推理
        result = inferencer(inputs=dict(points=bin_path), no_save_vis=True, no_save_pred=True)
        preds = result['predictions'][0]
        seg_labels = preds.get('pred_sem_seg', preds)['pts_semantic_mask']
        
        # 【核心修复】：统一转换为 NumPy 数组
        if hasattr(seg_labels, 'cpu'):
            seg_labels = seg_labels.cpu().numpy()
        else:
            seg_labels = np.array(seg_labels)
            
        np.save(out_path, seg_labels.astype(np.uint8))
        
    print("\n[+] All semantic masks successfully generated and saved to:", args.out_dir)

if __name__ == '__main__':
    main()