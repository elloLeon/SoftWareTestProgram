import torch
import time
import numpy as np
from sklearn.metrics import classification_report


def evaluate_model(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()  # 设置为评估模式

    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加速运行
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

### 准确性测试 ###
def evaluate_accuracy(model, data_loader, device='cpu'):
    """
    计算模型在给定数据集上的准确性。

    Args:
        model (torch.nn.Module): 已加载的模型。
        data_loader (torch.utils.data.DataLoader): 测试数据加载器。
        device (str): 使用的设备 ('cpu' 或 'cuda')。

    Returns:
        float: 准确率。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


### 性能测试 ###
def evaluate_performance(model, data_loader, device='cpu'):
    """
    评估模型的推理性能（时间）。

    Args:
        model (torch.nn.Module): 已加载的模型。
        data_loader (torch.utils.data.DataLoader): 测试数据加载器。
        device (str): 使用的设备 ('cpu' 或 'cuda')。

    Returns:
        float: 平均推理时间（每个样本）。
    """
    model.eval()
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_samples += images.size(0)

    return total_time / num_samples


### 公平性测试 ###
def evaluate_fairness(model, data_loader, device='cpu', sensitive_feature_indices=None):
    """
    检测模型是否对特定的敏感特征（如性别、种族）存在偏差。

    Args:
        model (torch.nn.Module): 已加载的模型。
        data_loader (torch.utils.data.DataLoader): 测试数据加载器。
        device (str): 使用的设备 ('cpu' 或 'cuda')。
        sensitive_feature_indices (list): 敏感特征的类别索引列表（如 [0, 1] 表示男性和女性）。

    Returns:
        dict: 每个敏感类别的分类报告。
    """
    model.to(device)  # 确保模型加载到指定设备
    model.eval()
    reports = {}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # 确保输入数据加载到设备
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            if sensitive_feature_indices:
                for idx in sensitive_feature_indices:
                    mask = labels == idx
                    labels_filtered = labels[mask]
                    predicted_filtered = predicted[mask]
                    reports[idx] = classification_report(
                        labels_filtered.cpu().numpy(),
                        predicted_filtered.cpu().numpy(),
                        output_dict=True
                    )

    return reports


### 蜕变测试 ###
def metamorphic_test(model, original_loader, transformed_loader, device='cpu'):
    """
    验证模型在数据变换前后的输出一致性。

    Args:
        model (torch.nn.Module): 已加载的模型。
        original_loader (torch.utils.data.DataLoader): 原始数据加载器。
        transformed_loader (torch.utils.data.DataLoader): 变换后数据加载器。
        device (str): 使用的设备 ('cpu' 或 'cuda')。

    Returns:
        float: 输出的一致性比例（0~1）。
    """
    model.eval()
    total_samples = 0
    consistent_samples = 0

    with torch.no_grad():
        for (original_images, _), (transformed_images, _) in zip(original_loader, transformed_loader):
            original_images, transformed_images = original_images.to(device), transformed_images.to(device)

            original_outputs = model(original_images)
            transformed_outputs = model(transformed_images)

            original_predictions = torch.argmax(original_outputs, dim=1)
            transformed_predictions = torch.argmax(transformed_outputs, dim=1)

            consistent_samples += (original_predictions == transformed_predictions).sum().item()
            total_samples += original_images.size(0)

    return consistent_samples / total_samples


