import torch.nn as nn
import torchvision.models as models


def create_model(num_classes):
    # 使用预训练的ResNet18作为基础模型
    model = models.resnet18(pretrained=True)

    # 修改最后的全连接层以适应我们的分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model