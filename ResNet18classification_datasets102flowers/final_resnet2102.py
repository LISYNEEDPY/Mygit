#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import torch.optim.lr_scheduler as lr_scheduler
import random
import matplotlib.pyplot as plt
# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 数据预处理和增强
data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(256),#随机中心裁剪为大小为256*256的图片
        transforms.RandomHorizontalFlip(),#随机翻转，默认概论0.5
        transforms.Resize(224),#随机裁剪为大小为224的图片
        transforms.RandomRotation(30),#随机角度旋转30度
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))#数据标准化加快模型收敛
    ]),
    'test':transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

}
# 加载数据集
data_dir = 'C:/Users/lisyneedpy'  # 数据集路径
image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'test']}

# 加载预训练的ResNet-18模型
resnet = resnet18(pretrained=True)
num_classes = 102  # 102种花朵类别
#print(resnet)
# 冻结所有模型参数
for param in resnet.parameters():
    param.requires_grad = False

# 解冻最后几个卷积层


for module in resnet.layer2.modules():
    for param in module.parameters():
        param.requires_grad = True
for module in resnet.layer3.modules():
    for param in module.parameters():
        param.requires_grad = True
for module in resnet.layer4.modules():
    for param in module.parameters():
        param.requires_grad = True

        # 替换最后的全连接层
resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
# 创建学习率调度器
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, verbose=True)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs=300):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        num.append(epoch)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
           
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'test':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
    print(f'\nBest Test Accuracy: {best_acc:.4f}')

# 训练模型
train_model(resnet, criterion, optimizer, num_epochs=10)#训练10个epoch