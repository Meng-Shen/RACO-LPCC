import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ====================== 工具函数 ======================
def load_point_cloud(bin_path):
    """读取原始 KITTI 格式的点云 (.bin)"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # 只取 xyz

def load_openpcdet_boxes(pkl_path, frame_id, score_thresh=0.3):
    """从OpenPCDet生成的 result.pkl 中读取对应帧的预测框"""
    if not os.path.exists(pkl_path):
        print(f"⚠️ 找不到预测结果文件: {pkl_path}")
        return np.empty((0, 7))
        
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
        
    # 遍历寻找对应帧 (通常 frame_id 是字符串，如 '000008')
    for res in results:
        if str(res['frame_id']) == str(frame_id):
            # 获取所有框和分数
            boxes = res['boxes_lidar']  # (N, 7) [x, y, z, dx, dy, dz, heading]
            scores = res['score']       # (N, )
            
            # 过滤掉低分预测框
            valid_mask = scores > score_thresh
            return boxes[valid_mask]
            
    print(f"⚠️ 在 pkl 中未找到帧 {frame_id} 的结果！")
    return np.empty((0, 7))

def draw_bev_boxes(ax, boxes, color='red', linewidth=1.5):
    """在BEV鸟瞰图上绘制3D旋转框的2D投影"""
    for box in boxes:
        x, y, z, dx, dy, dz, heading = box[:7]
        
        # 旋转矩阵 (绕Z轴/BEV视角下的平面旋转)
        c = np.cos(heading)
        s = np.sin(heading)
        R = np.array([
            [c, -s],
            [s,  c]
        ])
        
        # 矩形框相对于中心的四个角点
        corners = np.array([
            [ dx/2,  dy/2],
            [ dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2,  dy/2],
            [ dx/2,  dy/2]  # 回到起点以闭合多边形
        ])
        
        # 将角点旋转并平移到绝对坐标 (x, y)
        corners_rot = np.dot(corners, R.T)
        corners_rot[:, 0] += x
        corners_rot[:, 1] += y
        
        # 绘制该框的边界线
        ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=linewidth)

def visualize_and_save(points, pred_boxes, save_path):
    x = points[:, 0]
    y = points[:, 1]
    
    plt.figure(figsize=(16, 14), dpi=150)
    ax = plt.gca()
    
    # 1. 绘制纯黑色的点云
    # c='black' 保证点是黑色的，alpha=0.8 让点稍微有一点点透明度，显得不那么死板
    ax.scatter(x, y, c='black', s=0.05, alpha=0.8)
    
    # 2. 绘制预测框 (红色框在黑白背景下最显眼)
    if pred_boxes is not None and len(pred_boxes) > 0:
        draw_bev_boxes(ax, pred_boxes, color='red', linewidth=1.5)
        
    plt.axis('off')
    plt.tight_layout()
    # facecolor='white' 保证背景是纯白色的
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"✅ 保存成功: {save_path}")

# ====================== 命令行输入 ======================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法:")
        print(" python visualize_raw_boxes.py <点云.bin> <预测结果.pkl> <帧号>")
        print("例如:")
        print(" python visualize_raw_boxes.py 000008.bin result.pkl 000008")
        sys.exit(1)

    bin_file = sys.argv[1]
    pkl_file = sys.argv[2]
    frame_id = sys.argv[3]  # 如 '000008'
    
    # 加载数据 (可以根据需要调整 score_thresh 来过滤掉置信度低的框)
    pred_boxes = load_openpcdet_boxes(pkl_file, frame_id, score_thresh=0.4)
    points = load_point_cloud(bin_file)
    
    # 自动生成保存路径
    img_path = frame_id + "_det_boxes.png"

    visualize_and_save(points, pred_boxes, img_path)