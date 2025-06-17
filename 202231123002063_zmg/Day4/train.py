import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Vit_model import ViT


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


if __name__ == '__main__':
    # 使用示例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    img = torch.randn(1, 3, 256, 256).to(device)
    preds = model(img)

    print(preds.shape)  # torch.Size([1, 1000])