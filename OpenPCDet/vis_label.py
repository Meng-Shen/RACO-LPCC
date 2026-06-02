import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ====================== 工具函数 ======================
def load_point_cloud(bin_path):
    """读取原始 KITTI 格式的点云 (.bin)"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # 只取 xyz

def load_kitti_calib(calib_path):
    """
    加载KITTI标定文件，获取 Tr_velo_to_cam 雷达 -> 相机 变换矩阵
    我们需要逆矩阵：相机 -> 雷达
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    Tr_velo_to_cam = np.eye(4)
    for line in lines:
        if line.startswith('Tr_velo_to_cam:'):
            vals = np.array(line.strip().split()[1:], dtype=np.float32)
            Tr_velo_to_cam[:3, :4] = vals.reshape(3, 4)

    # 求逆矩阵：camera -> lidar
    T_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    return T_cam_to_velo

def cam_to_lidar(pts_3d, T_cam_to_velo):
    """把相机坐标系的3D点 转到 激光雷达坐标系"""
    pts_3d = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])  # 齐次坐标
    pts_lidar = np.dot(pts_3d, T_cam_to_velo.T)
    return pts_lidar[:, :3]

def load_kitti_label_boxes(label_path, calib_path, classes=['Car', 'Pedestrian', 'Cyclist']):
    """
    从 KITTI label_2 读取真实3D框，并转到激光雷达坐标系
    返回：N x 7 [x, y, z, l, w, h, ry] (LiDAR坐标系)
    """
    if not os.path.exists(label_path):
        print(f"⚠️ 找不到标签文件: {label_path}")
        return np.empty((0, 7))
    if not os.path.exists(calib_path):
        print(f"⚠️ 找不到标定文件: {calib_path}")
        return np.empty((0, 7))

    T_cam2velo = load_kitti_calib(calib_path)
    boxes = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        cls = parts[0]
        if cls not in classes: continue

        # KITTI Label 格式
        h = float(parts[8])   # 高度
        w = float(parts[9])   # 宽度
        l = float(parts[10])  # 长度
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        ry = float(parts[14])

        # 把中心点从相机坐标系 → 激光雷达坐标系
        center_cam = np.array([[x, y, z]])
        center_lidar = cam_to_lidar(center_cam, T_cam2velo)[0]

        # 旋转角修正（KITTI 相机系 ry → 雷达系 heading）
        heading = -ry - np.pi / 2

        boxes.append([
            center_lidar[0],
            center_lidar[1],
            center_lidar[2],
            l, w, h, heading
        ])

    return np.array(boxes, dtype=np.float32)

def draw_bev_boxes(ax, boxes, color='red', linewidth=1.5):
    """在BEV鸟瞰图上绘制3D旋转框的2D投影"""
    for box in boxes:
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

        ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=linewidth)


# 👉 修改点 1：增加 x_range 和 y_range 参数，并在画图前过滤和限制视野
def visualize_and_save(points, gt_boxes, save_path, x_range=None, y_range=None):
    
    # 🌟 优化：只保留指定范围内的点云数据
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

    if gt_boxes is not None and len(gt_boxes) > 0:
        draw_bev_boxes(ax, gt_boxes, color='red', linewidth=1.5)

    # 🌟 修改点 2：硬性限制画布的显示范围，进行裁剪
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
    if len(sys.argv) != 5:
        print("使用方法（必须传入标定文件！）:")
        print(" python vis_label.py <点云.bin> <标签.txt> <标定.txt> <帧号>")
        print("例子:")
        print(" python vis_label.py 000008.bin label_2/000008.txt calib/000008.txt 000008")
        sys.exit(1)

    bin_file = sys.argv[1]
    label_file = sys.argv[2]
    calib_file = sys.argv[3]
    frame_id = sys.argv[4]

    points = load_point_cloud(bin_file)
    gt_boxes = load_kitti_label_boxes(label_file, calib_file)

    # 👇👇👇 在这里手动输入你想要的 X 和 Y 坐标范围 👇👇👇
    X_RANGE = None#(0, 10)  # 例如 X轴(前后) -20米 到 20米
    Y_RANGE = (0, 30)  # 例如 Y轴(左右) -20米 到 20米
    # 如果某一个轴不想限制，可以设置为 None，例如：Y_RANGE = None
    # 👆👆👆 -------------------------------------- 👆👆👆

    img_path = frame_id + "_gt_aligned_cropped.png" # 修改了默认的文件名，避免覆盖原图
    
    # 将坐标范围参数传进去
    visualize_and_save(points, gt_boxes, img_path, x_range=X_RANGE, y_range=Y_RANGE)