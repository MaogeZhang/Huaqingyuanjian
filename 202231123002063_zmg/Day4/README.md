# Vision Transformer (ViT) 实现说明

## 架构概述
1. **Patch Embedding**:
   - 将输入图像分割为固定大小的patch
   - 线性投影到模型维度

2. **Transformer Encoder**:
   - 多头自注意力机制 (Multi-Head Attention)
   - 前馈网络 (Feed Forward Network)
   - 残差连接和层归一化

3. **分类头**:
   - 使用CLS token或全局平均池化
   - 线性投影到类别数

## 关键组件
- `FeedForward`: 两层MLP带GELU激活
- `Attention`: 多头自注意力机制
- `Transformer`: 堆叠的Transformer层
- `ViT`: 完整ViT模型

## 使用示例
```python
from vit import ViT

model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048
)