import time
import torch
import argparse
from mmdet3d.apis import LidarSeg3DInferencer

def parse_args():
    parser = argparse.ArgumentParser(description='Measure Inference Speed, Params, and Memory of 3D Seg Models')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('--pcd', default='demo/data/000008.bin', help='用于测试的点云文件')
    parser.add_argument('--iters', type=int, default=50, help='测试循环次数')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n=> 正在初始化模型: {args.config}")
    # 不传入 weights 参数，模型会自动随机初始化
    inferencer = LidarSeg3DInferencer(args.config, weights=None, device='cuda:0')
    
    # ---------------- 新增：计算模型参数量 (Params) ----------------
    # inferencer.model 获取底层 PyTorch 模型
    total_params = sum(p.numel() for p in inferencer.model.parameters())
    print(f"=> 模型参数量: {total_params / 1e6:.2f} M (百万)")
    # ---------------------------------------------------------------

    print("=> 正在进行 GPU 预热 (Warm-up 10 iterations)...")
    # 预热是为了让 GPU 频率拉满，并完成 CUDA 内存分配，否则第一次推理会特别慢
    for _ in range(10):
        # 推理但不打印或保存结果
        inferencer(dict(points=args.pcd), show=False)

    print(f"=> 开始测速 (测试 {args.iters} 帧)...")
    
    # ---------------- 新增：重置显存峰值统计 ----------------
    torch.cuda.reset_peak_memory_stats(device='cuda:0')
    # --------------------------------------------------------
    
    torch.cuda.synchronize()  # 确保所有之前的 GPU 任务都已完成
    start_time = time.perf_counter()

    for _ in range(args.iters):
        inferencer(dict(points=args.pcd), show=False)

    torch.cuda.synchronize()  # 等待所有推理任务完成
    end_time = time.perf_counter()

    # ---------------- 新增：获取最大显存占用 ----------------
    # 获取循环期间的峰值显存 (Bytes 转换为 MB)
    max_memory = torch.cuda.max_memory_allocated(device='cuda:0') / (1024 ** 2)
    # --------------------------------------------------------

    total_time = end_time - start_time
    avg_latency = (total_time / args.iters) * 1000  # 转为毫秒
    fps = 1.0 / (avg_latency / 1000.0)

    print("\n" + "="*50)
    print(f"模型配置:         {args.config.split('/')[-1]}")
    print(f"模型参数量 (Params): {total_params / 1e6:.2f} M")
    print(f"最大显存占用:     {max_memory:.2f} MB")
    print(f"总耗时:           {total_time:.4f} 秒")
    print(f"平均延迟 (Latency): {avg_latency:.2f} ms / 帧")
    print(f"吞吐量 (FPS):      {fps:.2f} 帧 / 秒")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()