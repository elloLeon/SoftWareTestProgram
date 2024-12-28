import torch
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def load_resnet():
    model = models.resnet18(pretrained=True)  # 选择预训练的 ResNet18 模型
    num_ftrs = model.fc.in_features  # 获取原模型的输出特征数
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 将最后一层输出改为10类，适配CIFAR-10

    model.eval()  # 设置为推理模式
    return model

def train_model(model, train_loader, epochs=5, lr=0.001, device='cpu'):
    """
    训练模型
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    model.train()  # 设置为训练模式

    for epoch in range(epochs):
        running_loss = 0.0

        # 使用 tqdm 包裹 train_loader，以便显示进度条
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, unit="batch") as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # 清空梯度
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                running_loss += loss.item()

                # 更新进度条描述
                t.set_postfix(loss=running_loss / len(t))

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    print("Finished Training")
    return model