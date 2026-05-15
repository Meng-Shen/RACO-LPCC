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
    """
    从OpenPCDet生成的 result.pkl 中读取：
    只保留 Car, Pedestrian, Cyclist
    并且过滤低置信度
    """
    if not os.path.exists(pkl_path):
        print(f"⚠️ 找不到预测结果文件: {pkl_path}")
        return np.empty((0, 7)), np.empty((0,))
        
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
        
    # 遍历寻找对应帧
    for res in results:
        if str(res['frame_id']) == str(frame_id):
            boxes = res['pred_boxes']    
            scores = res['pred_scores']  
            labels = res['pred_labels']   # 🔥 类别标签

            # ====================== 🔥 关键过滤 ======================
            # 只保留这 3 个类别：1:Car, 2:Pedestrian, 3:Cyclist
            keep_classes = [1,2,3]
            class_mask = np.isin(labels, keep_classes)
            
            # 置信度阈值
            score_mask = scores > score_thresh
            
            # 同时满足：属于3类 + 置信度够高
            valid_mask = class_mask & score_mask
            # =========================================================
            
            return boxes[valid_mask], scores[valid_mask]
            
    print(f"⚠️ 在 pkl 中未找到帧 {frame_id} 的结果！")
    return np.empty((0, 7)), np.empty((0,))

def score_to_color(score):
    """置信度 → 颜色：越低越红，越高越蓝"""
    r = 1.0 - score
    g = 0.0
    b = score
    return (r, g, b)

def draw_bev_boxes(ax, boxes, scores, linewidth=1.5):
    """根据置信度绘制渐变颜色框"""
    for box, score in zip(boxes, scores):
        x, y, z, dx, dy, dz, heading = box[:7]
        
        c = np.cos(heading)
        s = np.sin(heading)
        R = np.array([[c, -s],[s, c]])
        
        corners = np.array([
            [ dx/2,  dy/2],
            [ dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2,  dy/2],
            [ dx/2,  dy/2]
        ])
        
        corners_rot = np.dot(corners, R.T)
        corners_rot[:, 0] += x
        corners_rot[:, 1] += y
        
        color = score_to_color(score)
        ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=linewidth)

# 👉 修改点 1：新增 x_range 和 y_range 参数
def visualize_and_save(points, pred_boxes, pred_scores, save_path, x_range=None, y_range=None):
    
    # 🌟 优化：在画图前过滤掉不在范围内的点，加快渲染速度
    if x_range is not None:
        mask_x = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        points = points[mask_x]
    if y_range is not None:
        mask_y = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        points = points[mask_y]

    x = points[:, 0]
    y = points[:, 1]
    
    plt.figure(figsize=(16, 14), dpi=150)
    ax = plt.gca()
    
    ax.scatter(x, y, c='black', s=1, alpha=0.8)
    
    if pred_boxes is not None and len(pred_boxes) > 0:
        i = 1
        draw_bev_boxes(ax, pred_boxes, pred_scores, linewidth=1.5)
        
    # 🌟 修改点 2：硬性限制画布的显示范围，避免被边缘的框拉大视野
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"✅ 保存成功: {save_path}")

# ====================== 命令行输入 ======================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法:")
        print(" python visualize_raw_boxes.py <点云.bin或npy> <预测结果.pkl> <帧号>")
        print("例如:")
        print(" python visualize_raw_boxes.py 000008.npy result.pkl 000008")
        sys.exit(1)

    npy_file = sys.argv[1]
    pkl_file = sys.argv[2]
    frame_id = sys.argv[3]
    
    # 只保留 车、人、自行车，置信度 >0.4
    pred_boxes, pred_scores = load_openpcdet_boxes(pkl_file, frame_id, score_thresh=0.4)
    
    if npy_file.endswith('.npy'):
        points = np.load(npy_file)[:, :3]
    else:
        points = load_point_cloud(npy_file)
    
    # 👇👇👇 在这里手动输入你想要的 X 和 Y 坐标范围 👇👇👇
    X_RANGE = None#(0, 10)  # 例如前后 -20米 到 20米
    Y_RANGE = (0, 30)  # 例如左右 -20米 到 20米
    # 如果某一个轴不想限制，可以设置为 None，例如：Y_RANGE = None
    # 👆👆👆 -------------------------------------- 👆👆👆

    img_path = frame_id + "_det_boxes_cropped.png"
    
    # 传入范围参数
    visualize_and_save(points, pred_boxes, pred_scores, img_path, x_range=X_RANGE, y_range=Y_RANGE)