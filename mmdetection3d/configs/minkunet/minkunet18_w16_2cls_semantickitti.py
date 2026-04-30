_base_ = ['./minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti.py']

# ================= 1. 标签映射设置 (动态生成) =================

# 步骤 A: 照搬 SemanticKITTI 官方的 raw (250+) 到 19 分类的标准映射表
standard_learning_map = {
    0: 255, 1: 255, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4,
    30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13,
    52: 255, 60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 255,
    252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4
}

# 步骤 B: 按照你的需求，指定 19 分类中的哪些索引属于"前景"
fg_19_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

# 步骤 C: 动态生成 200+ 到 2分类 的终极映射字典
label_mapping = {i: 255 for i in range(260)} # 初始化：0~259 全部设为忽略(255)

for raw_label, cls_19 in standard_learning_map.items():
    if cls_19 == 255:
        label_mapping[raw_label] = 255 # 官方要求忽略的，继续忽略
    elif cls_19 in fg_19_indices:
        label_mapping[raw_label] = 1   # 命中你的列表，设为前景 (1)
    else:
        label_mapping[raw_label] = 0   # 否则，设为背景 (0)

# 将动态生成的 label_mapping 塞给 metainfo
metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[128, 128, 128], [255, 0, 0]],  # 背景灰色，前景红色
    seg_label_mapping=label_mapping,
    max_label=259
)

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

# ================= 2. 模型轻量化与损失函数修改 =================
# ================= 2. 模型轻量化与损失函数修改 =================
model = dict(
    backbone=dict(
        # 删掉错误的 type='MinkUNet18'，让它自动继承 base 中的类名
        # 通过修改 encoder_blocks 把 34 层降为 18 层
        base_channels=16,
        encoder_channels=[16, 32, 64, 128],
        encoder_blocks=[2, 2, 2, 2],  # 原本是 [3, 4, 6, 3] 代表 34 层，现在改成 [2, 2, 2, 2] 变 18 层
        decoder_channels=[128, 64, 32, 16],
        decoder_blocks=[1, 1, 1, 1]
    ),
    decode_head=dict(
        channels=16,        # 与 backbone 输出对齐
        num_classes=2,      # 变为 2 分类
        ignore_index=255,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, 10.0],  # 前景分错十倍惩罚
            loss_weight=1.0
        )
    )
)