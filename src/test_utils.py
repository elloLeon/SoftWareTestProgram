import torch


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
