import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择模型：resnet18, googlenet, mobilenet_v2, moganet (需单独安装)
model_name = 'googlenet'  # 可修改为 'googlenet', 'mobilenet_v2', 'moganet'


def get_model(name, num_classes):
    if name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif name == 'moganet':
        from timm import create_model  # pip install timm
        model = create_model('moganet_s', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name.")
    return model.to(device)


net = get_model(model_name, num_classes=10)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 兼容大多数模型输入
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# 训练函数
def train(epoch):
    net.train()
    total_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
