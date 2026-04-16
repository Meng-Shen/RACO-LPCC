import pandas as pd

def count_scales_from_csv(csv_path):
    """
    从本地 CSV 文件读取并统计
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 按 jucp_label 和 scale 分组并统计数量
        counts = df.groupby(['jucp_label', 'scale']).size().reset_index(name='count')
        
        # 按照 jucp_label 从小到大（即压缩率从低到高）排序
        counts = counts.sort_values('jucp_label')
        
        print(f"========== 文件 {csv_path} 统计结果 ==========")
        print(f"{'Label':<8} | {'Scale (量化步长)':<20} | {'Count (帧数)'}")
        print("-" * 50)
        
        total_frames = 0
        for index, row in counts.iterrows():
            label = int(row['jucp_label'])
            scale = row['scale']
            count = int(row['count'])
            total_frames += count
            print(f"{label:<8} | {scale:<20} | {count}")
            
        print("-" * 50)
        print(f"总计: {total_frames} 帧\n")
        
    except FileNotFoundError:
        print(f"找不到文件: {csv_path}，请检查路径。")

if __name__ == '__main__':
    # 假设你的文件名为 jucp_labels.csv，和这个脚本放在同一个目录下
    # 你可以修改这里的路径指向你的实际文件
    file_path = 'jucp_labels.csv' 
    
    count_scales_from_csv(file_path)