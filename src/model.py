import torch
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os

def load_resnet():
    """
    加载 ResNet18 模型，并修改输出层以适配 CIFAR-10 数据集。
    """
    model = models.resnet18(pretrained=True)  # 选择预训练的 ResNet18 模型
    num_ftrs = model.fc.in_features  # 获取原模型的输出特征数
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 将最后一层输出改为10类，适配CIFAR-10

    model.eval()  # 设置为推理模式
    return model

def train_model(model, train_loader, epochs=5, lr=0.001, device='cpu', save_path=None):
    """
    训练模型，并根据需要保存训练好的模型。

    Args:
        model (torch.nn.Module): 模型实例。
        train_loader (DataLoader): 训练数据加载器。
        epochs (int): 训练轮次。
        lr (float): 学习率。
        device (str): 设备 ('cpu' 或 'cuda')。
        save_path (str): 保存模型的路径（可选）。
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

    # 如果提供了保存路径，则保存模型
    if save_path:
        save_model(model, save_path)

    return model

def save_model(model, path="resnet_model.pth"):
    """
    保存模型到本地文件。

    Args:
        model (torch.nn.Module): 要保存的模型。
        path (str): 模型保存路径。
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="resnet_model.pth", device="cpu"):
    """
    从本地文件加载模型。

    Args:
        model (torch.nn.Module): 要加载的模型结构（未初始化参数）。
        path (str): 模型保存路径。
        device (str): 加载模型的设备 ('cpu' 或 'cuda')。

    Returns:
        torch.nn.Module: 加载了参数的模型。
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}. Please train the model first.")
    return model