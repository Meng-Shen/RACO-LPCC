import os
import argparse
import csv
import time
import numpy as np
from tqdm import tqdm
from mmdet3d.apis import inference_segmentor, init_model
try:
    import torch
except ImportError:
    torch = None

def main():
    parser = argparse.ArgumentParser(description="Offline Semantic Mask Generation")
    parser.add_argument('--val_txt', type=str, default='../data/kitti/ImageSets/train.txt', help='Path to val.txt')
    parser.add_argument('--bin_dir', type=str, default='../data/kitti/training/velodyne', help='Path to velodyne bins')
    parser.add_argument('--out_dir', type=str, default='../output/eval/train_seg_masks', help='Output directory for .npy masks')
    
    # 必须传入 mmdet3d 的配置和权重
    parser.add_argument('--seg_cfg_file', type=str, required=True)
    parser.add_argument('--seg_ckpt', type=str, required=True)
    parser.add_argument('--time_csv', type=str, default=None, help='Optional CSV used to append per-frame segmentation time')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for segmentation inference')
    parser.add_argument(
        '--fg_threshold',
        type=float,
        default=0.35,
        help='Foreground probability threshold. Lower values increase foreground recall at the cost of more background false positives.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.time_csv:
        time_dir = os.path.dirname(args.time_csv)
        if time_dir:
            os.makedirs(time_dir, exist_ok=True)

    with open(args.val_txt, 'r') as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]

    pending_ids = [
        vid for vid in val_ids
        if not os.path.exists(os.path.join(args.out_dir, f"{vid}.npy"))
    ]

    if not pending_ids:
        print("\n[+] All semantic masks already exist, skip segmentation:", args.out_dir)
        return

    print(f"[*] Loading Semantic Model from {args.seg_ckpt}...")
    model = init_model(args.seg_cfg_file, args.seg_ckpt, device=args.device)

    time_file = None
    time_writer = None
    if args.time_csv:
        file_exists = os.path.exists(args.time_csv) and os.path.getsize(args.time_csv) > 0
        time_file = open(args.time_csv, 'a', newline='')
        time_writer = csv.DictWriter(time_file, fieldnames=['frame_id', 'seg_time'])
        if not file_exists:
            time_writer.writeheader()

    for vid in tqdm(val_ids, desc="Generating Masks"):
        out_path = os.path.join(args.out_dir, f"{vid}.npy")
        if os.path.exists(out_path):
            continue
            
        bin_path = os.path.join(args.bin_dir, f"{vid}.bin")
        
        # 纯净的单帧推理
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        result, _ = inference_segmentor(model, bin_path)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        seg_time = time.time() - start_time
        if hasattr(result, 'pts_seg_logits') and hasattr(result.pts_seg_logits, 'pts_seg_logits'):
            seg_logits = result.pts_seg_logits.pts_seg_logits
            if torch is not None and hasattr(seg_logits, 'softmax'):
                fg_prob = seg_logits.softmax(dim=0)[1]
                seg_labels = (fg_prob >= args.fg_threshold).to(torch.uint8)
            else:
                logits = np.array(seg_logits)
                logits = logits - logits.max(axis=0, keepdims=True)
                prob = np.exp(logits)
                prob = prob / prob.sum(axis=0, keepdims=True)
                seg_labels = (prob[1] >= args.fg_threshold).astype(np.uint8)
        else:
            seg_labels = result.pred_pts_seg.pts_semantic_mask
        
        # 【核心修复】：统一转换为 NumPy 数组
        if hasattr(seg_labels, 'cpu'):
            seg_labels = seg_labels.cpu().numpy()
        else:
            seg_labels = np.array(seg_labels)
            
        np.save(out_path, seg_labels.astype(np.uint8))
        if time_writer is not None:
            time_writer.writerow({'frame_id': vid, 'seg_time': f'{seg_time:.6f}'})
            time_file.flush()

    if time_file is not None:
        time_file.close()
        
    print("\n[+] All semantic masks successfully generated and saved to:", args.out_dir)

if __name__ == '__main__':
    main()
