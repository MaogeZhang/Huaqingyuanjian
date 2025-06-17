关键模块说明表格
模块	类型	功能说明	典型参数
Conv	卷积层	基础卷积块（Conv+BN+SiLU）	[channels, kernel_size, stride]
C2f	特征提取	改进的CSP结构，带残差连接	[channels, shortcut]
SPPF	池化层	快速空间金字塔池化	[channels, kernel_size]
Concat	特征融合	多尺度特征拼接	[dimension]
Upsample	上采样	特征图分辨率提升	[scale_factor, mode]
Detect	检测头	多尺度预测输出	[num_classes]
# YOLOv8n模型配置示例
scales:
  n: [0.33, 0.25, 1024]  # depth, width, max_channels

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]    # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]   # 1-P2/4
  - [-1, 3, C2f, [128, True]]    # 2
  - [-1, 1, Conv, [256, 3, 2]]   # 3-P3/8
  - [-1, 6, C2f, [256, True]]    # 4
  - [-1, 1, Conv, [512, 3, 2]]   # 5-P4/16
  - [-1, 6, C2f, [512, True]]    # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]   # 8
  - [-1, 1, SPPF, [1024, 5]]     # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                   # 11
  - [-1, 3, C2f, [512, False]]                  # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 4], 1, Concat, [1]]                   # 14
  - [-1, 3, C2f, [256, False]]                  # 15
  - [-1, 1, Conv, [256, 3, 2]]                  # 16
  - [[-1, 12], 1, Concat, [1]]                  # 17
  - [-1, 3, C2f, [512, False]]                  # 18
  - [-1, 1, Conv, [512, 3, 2]]                  # 19
  - [[-1, 9], 1, Concat, [1]]                   # 20
  - [-1, 3, C2f, [1024, False]]                 # 21
  - [[15, 18, 21], 1, Detect, [nc]]             # 22