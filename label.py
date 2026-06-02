import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 移除了中文字体设置，直接使用 matplotlib 默认的英文字体

def generate_semantickitti_fg_bg_map():
    # 1. Define inputs and definitions
    fg_classes_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]
    
    class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]

    # Standard SemanticKITTI class definitions
    class_def = {
        0: {"name": class_names[0]},
        1: {"name": class_names[1]},
        2: {"name": class_names[2]},
        3: {"name": class_names[3]},
        4: {"name": class_names[4]},
        5: {"name": class_names[5]},
        6: {"name": class_names[6]},
        7: {"name": class_names[7]},
        8: {"name": class_names[8]},
        9: {"name": class_names[9]},
        10: {"name": class_names[10]},
        11: {"name": class_names[11]},
        12: {"name": class_names[12]},
        13: {"name": class_names[13]},
        14: {"name": class_names[14]},
        15: {"name": class_names[15]},
        16: {"name": class_names[16]},
        17: {"name": class_names[17]},
        18: {"name": class_names[18]}
    }

    all_raw_indices = sorted(list(set(fg_classes_indices + [8, 9, 10, 11, 12, 13, 14, 15, 16])))

    final_mapping = {}
    for i in all_raw_indices:
        if i in fg_classes_indices:
            final_mapping[i] = "FG"
        else:
            final_mapping[i] = "BG"

    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    y_range = len(all_raw_indices) * 1.2 + 5
    ax.set_ylim(0, y_range)
    ax.axis('off') 
    plt.title("Mapping SemanticKITTI Raw IDs to Binary FG/BG", fontsize=16, fontweight='bold', pad=20)

    def draw_box(ax, x, y, w, h, text, color, text_color='black', fontsize=11, fontweight='normal'):
        rect = patches.FancyBboxPatch((x, y), w, h, 
                                      linewidth=1.0, 
                                      edgecolor='black', 
                                      facecolor=color, 
                                      boxstyle='round,pad=0.1')
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color=text_color, fontweight=fontweight)

    # Colors
    color_raw = '#f5f5f5' 
    color_fg = '#e74c3c'  
    color_bg = '#3498db'  

    num_classes = len(all_raw_indices)
    h_item = 0.8
    gap = 0.4
    col1_x = 0.5
    col1_w = 2.8
    
    y_top = y_range - 3
    y_current = y_top

    # --- COLUMN 1: Raw Semantic Indices ---
    ax.text(col1_x + col1_w/2, y_top + 1.5, "Raw Semantic Indices", ha='center', va='center', fontsize=12, fontweight='bold')
    #ax.text(col1_x + col1_w/2, y_top + 0.6, "ID : Physical Meaning", ha='center', va='center', fontsize=10, style='italic')

    raw_boxes_ports = {} 
    
    for idx in all_raw_indices:
        name_info = class_def.get(idx, {"name": f"unknown_{idx}"})
        name = name_info["name"].capitalize()
        text = f"{idx} : {name}"
        
        draw_box(ax, col1_x, y_current, col1_w, h_item, text, color_raw)
        raw_boxes_ports[idx] = (col1_x + col1_w, y_current + h_item/2)
        y_current -= (h_item + gap)

    # --- COLUMN 2: Binary Mapping ---
    col2_x = 5.0
    col2_w = 1.2
    y_targets_center = (y_top + y_current + h_item + gap) / 2 

    ax.text(col2_x + col2_w/2, y_targets_center + 3.0, "Binary Mapping", ha='center', va='center', fontsize=12, fontweight='bold')
    #ax.text(col2_x + col2_w/2, y_targets_center + 2.1, "1=FG, 0=BG", ha='center', va='center', fontsize=10, style='italic')

    h_target = 2.0
    gap_target = 1.0
    y_fg_target = y_targets_center + gap_target/2
    y_bg_target = y_targets_center - h_target - gap_target/2

    draw_box(ax, col2_x, y_fg_target, col2_w, h_target, "Foreground", color_fg, text_color='white', fontweight='bold', fontsize=12)
    draw_box(ax, col2_x, y_bg_target, col2_w, h_target, "Background", color_bg, text_color='white', fontweight='bold', fontsize=12)

    port_fg_target = (col2_x, y_fg_target + h_target/2)
    port_bg_target = (col2_x, y_bg_target + h_target/2)

    # --- Draw arrows ---
    fg_count = 0
    bg_count = 0
    num_fg = sum(1 for v in final_mapping.values() if v == "FG")
    num_bg = sum(1 for v in final_mapping.values() if v == "BG")

    for idx in all_raw_indices:
        start_pt = raw_boxes_ports[idx]
        target_group = final_mapping[idx]
        
        if target_group == "FG":
            offset = (fg_count - num_fg/2 + 0.5) * 0.15
            end_pt = (port_fg_target[0], port_fg_target[1] + offset)
            color_arrow = color_fg
            fg_count += 1
        else:
            offset = (bg_count - num_bg/2 + 0.5) * 0.15
            end_pt = (port_bg_target[0], port_bg_target[1] + offset)
            color_arrow = color_bg
            bg_count += 1
            
        ax.annotate("", xy=end_pt, xytext=start_pt, 
                    arrowprops=dict(arrowstyle="->", 
                                    color=color_arrow, 
                                    linewidth=1.0, 
                                    connectionstyle="arc3,rad=0.1")) 

    # --- COLUMN 3: FG/BG Logical View ---
    col3_x = 7.8
    col3_w = 2.0
    ax.text(col3_x + col3_w/2, y_top + 1.5, "3. Logical View", ha='center', va='center', fontsize=12, fontweight='bold')

    fg_ids_text = ",\n".join([f"• {idx}" for idx in fg_classes_indices])
    bg_ids_text = "• All other IDs\n  (e.g., 9, 10, 11...)"

    y_fg_logic = y_fg_target + 0.2
    draw_box(ax, col3_x, y_fg_logic, col3_w, 1.6, f"Foreground IDs:\n\n{fg_ids_text}", color_raw, fontsize=10)

    y_bg_logic = y_bg_target + 0.2
    draw_box(ax, col3_x, y_bg_logic, col3_w, 1.6, f"Background IDs:\n\n{bg_ids_text}", color_raw, fontsize=10)

    ax.text(5, 1.0, f"* FG_CLASSES defined as: {fg_classes_indices}", ha='center', fontsize=9, style='italic', color='gray')
    ax.text(5, 0.4, "* Note: Only a representative subset of IDs is shown above.", ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    return fig

try:
    fig_semantickitti_mapping = generate_semantickitti_fg_bg_map()
    # 因为你在纯命令行 Linux 上跑，可能没有图形界面弹窗 (X11)
    # 强烈建议直接保存为图片，而不是 plt.show()
    fig_semantickitti_mapping.savefig('semantic_mapping.png', dpi=300, bbox_inches='tight')
    print("Success! Image saved as 'semantic_mapping.png' in the current directory.")
except Exception as e:
    print(f"Error occurred: {e}")