import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing

# =========================================================
# 子进程专属全局变量：每个进程独立持有一份数据，互不干扰
# =========================================================
W_GT_ANNOS = None
W_DT_ANNOS_ALL = None
W_BASELINE_IDX = 0  # 0 对应 Label 0 (最大码率/原图)
W_CLASSES = ['Car', 'Pedestrian', 'Cyclist']

def init_worker(cfg_file, split_file, eval_dir):
    """
    进程初始化函数：子进程被 spawn 出来后，第一件事就是独立读取数据。
    在此函数内加载 PyTorch，保证 CUDA Context 是子进程专有的。
    """
    global W_GT_ANNOS, W_DT_ANNOS_ALL

    # 屏蔽 OpenPCDet 数据集初始化时的海量打印
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull

    try:
        import _init_path
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader

        cfg_from_yaml_file(cfg_file, cfg)

        # 独立加载真实标签 (GT)
        dataset, _, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
            dist=False, workers=1, training=False, logger=None
        )
        W_GT_ANNOS = [info['annos'] for info in dataset.kitti_infos]

        # 严格遵照之前生成 CSV 时的目录顺序
        original_quant_map = [
            (1.5/256, 1/512), # combo 0 (最强压缩)
            (2/256, 1/512),   # combo 1
            (3/256, 1/512),   # combo 2
            (4/256, 1/512),   # combo 3
            (1/64, 1.25/512), # combo 4
            (1/64, 1.5/512)   # combo 5 (最大码率/原图)
        ]

        eval_dir_path = Path(eval_dir)
        W_DT_ANNOS_ALL = {}

        for combo_idx in range(6):
            scale_fg, scale_bg = original_quant_map[combo_idx]
            folder_name = f'combo_{combo_idx}_fg_{scale_fg:.6f}_bg_{scale_bg:.6f}'
            pkl_path = eval_dir_path / folder_name / 'result.pkl'
            
            # 【关键映射】：让 Label 0 = 原图(combo_5), Label 5 = 最强压缩(combo_0)
            target_label = 5 - combo_idx  
            with open(pkl_path, 'rb') as f:
                W_DT_ANNOS_ALL[target_label] = pickle.load(f)

    finally:
        sys.stdout = old_stdout
        devnull.close()

def evaluate_frame_worker(frame_idx):
    """
    评测函数：只单独降级第 frame_idx 帧，计算出 6 个码率下的全局 AP
    """
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    
    try:
        # 在 Worker 内部导入评测脚本，防止在外面污染环境
        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

        frame_id = W_DT_ANNOS_ALL[0][frame_idx]['frame_id']
        row_result = {'frame_idx': frame_idx, 'frame_id': frame_id}
        
        # 提取全集均为最大码率(Baseline)的预测结果拷贝
        base_dt_annos = list(W_DT_ANNOS_ALL[W_BASELINE_IDX])
        
        for b in range(6):
            if b == W_BASELINE_IDX:
                mixed_dt_annos = base_dt_annos
            else:
                mixed_dt_annos = base_dt_annos.copy()
                mixed_dt_annos[frame_idx] = W_DT_ANNOS_ALL[b][frame_idx]

            # 核心接口调用：计算全局 AP
            _, ap_dict = kitti_eval.get_official_eval_result(
                W_GT_ANNOS, mixed_dt_annos, W_CLASSES
            )

            # 提取 3D Moderate AP
            row_result[f'L{b}_Car_AP'] = float(ap_dict.get('Car_3d/moderate_R40', 0.0))
            row_result[f'L{b}_Ped_AP'] = float(ap_dict.get('Pedestrian_3d/moderate_R40', 0.0))
            row_result[f'L{b}_Cyc_AP'] = float(ap_dict.get('Cyclist_3d/moderate_R40', 0.0))
            
        return row_result

    except Exception as e:
        sys.stdout = old_stdout
        print(f"\nWorker Error at frame {frame_idx}: {e}")
        return None
    finally:
        sys.stdout = old_stdout
        devnull.close()

def parse_config():
    parser = argparse.ArgumentParser(description='Single Frame AP Drop Sensitivity Analysis')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to OpenPCDet model config')
    parser.add_argument('--split_file', type=str, required=True, help='Path to val.txt or train.txt')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to directory containing combo_* folders')
    parser.add_argument('--out_csv', type=str, default='jucp_ap_sensitivity.csv', help='Output CSV file name')
    parser.add_argument('--workers', type=int, default=8, help='Number of CPU cores to use. Use 4-8 is best.')
    args = parser.parse_args()
    return args

def main():
    # 强行设置多进程启动模式为 spawn，这是避免 CUDA Fork 死锁的唯一神技
    multiprocessing.set_start_method('spawn', force=True)

    args = parse_config()
    
    print("[1/3] Detecting number of frames...")
    
    # 我们只需读取原图(combo_5)的 pkl 来快速知道有几帧，主进程绝不碰 PyTorch
    scale_fg, scale_bg = 1/64, 1.5/512
    test_pkl = Path(args.eval_dir) / f'combo_5_fg_{scale_fg:.6f}_bg_{scale_bg:.6f}' / 'result.pkl'
    
    if not test_pkl.exists():
        print(f"❌ Error: Cannot find {test_pkl}")
        return
        
    with open(test_pkl, 'rb') as f:
        test_data = pickle.load(f)
        num_frames = len(test_data)
    del test_data

    print(f"      Detected {num_frames} frames.")

    print("[2/3] Firing up Independent Isolated Workers (Spawning)...")
    print("      Each worker will load its own Data & CUDA context independently.")
    print("      This safely prevents 'CUDA initialized before forking' errors.")
    
    num_workers = args.workers if args.workers > 0 else 8

    final_results = []
    
    # 启动多进程池 (Pool)
    with multiprocessing.Pool(
        num_workers, 
        initializer=init_worker, 
        initargs=(args.cfg_file, args.split_file, args.eval_dir)
    ) as pool:
        
        for res in tqdm(pool.imap_unordered(evaluate_frame_worker, range(num_frames)), total=num_frames, desc="Calculating global APs"):
            if res is not None:
                final_results.append(res)
                
                # ===== 【新增的代码：实时打印单帧结果】 =====
                # 提取一些关键指标进行展示，比如 Car 类在原图(L0)和最狠压缩(L5)下的 AP 对比
                frame_id = res['frame_id']
                l0_car = res['L0_Car_AP']
                l5_car = res['L5_Car_AP']
                
                # 如果你想看全部 18 个指标，可以改成 tqdm.write(str(res))
                # 这里为你格式化了一个非常直观的摘要输出
                msg = f"✅ 帧 {frame_id} 处理完毕 | 车辆AP: 原图(L0)={l0_car:.4f} ➔ 极限压缩(L5)={l5_car:.4f}"
                tqdm.write(msg)
                # ============================================

    print("[3/3] Saving Final AP Matrix to CSV...")
    df = pd.DataFrame(final_results)
    
    # 根据帧索引排好序，并丢弃临时辅导列 frame_idx
    df = df.sort_values('frame_idx').drop(columns=['frame_idx'])
    df.to_csv(args.out_csv, index=False)
    
    print(f"\n✨ Success! Global AP sensitivities safely saved to {args.out_csv}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

if __name__ == '__main__':
    main()