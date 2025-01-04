import os

from src.data import load_cifar10
from src.model import load_resnet, train_model, load_model, save_model
from src.transforms import add_noise, add_blur, add_flip,add_rotation
from src.test_utils import evaluate_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np

def test_model_accuracy_with_variations():
    """
    Tests the accuracy of the model with variations like noise, blur, and flips.
    性能测试: 检测模型在不同数据变异下的正确率
    """
    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 加载并训练模型
    model = load_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = train_model(model, train_loader, epochs=10, lr=0.001, device=device)

    model_path = "./data/resnet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)

    else:
        print("Training model...")
        model = train_model(model, train_loader, epochs=3, lr=0.001, device=device)
        save_model(model, path=model_path)  # 保存模型
        print(f"Model saved to {model_path}.")



    # 测试原始数据的准确性
    original_accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Original Test Accuracy: {original_accuracy * 100:.2f}%")

    # 对测试数据添加噪声并评估
    test_images, test_labels = next(iter(test_loader))
    noisy_images = add_noise(test_images.numpy(), noise_level=0.1)
    noisy_dataset = torch.utils.data.TensorDataset(
        torch.tensor(noisy_images), torch.tensor(test_labels)
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)
    noisy_accuracy = evaluate_accuracy(model, noisy_loader, device)
    print(f"Test Accuracy with Noise: {noisy_accuracy * 100:.2f}%")

    # 对测试数据添加模糊并评估
    blurred_images = add_blur(test_images.numpy(), kernel_size=5)
    blurred_dataset = torch.utils.data.TensorDataset(
        torch.tensor(blurred_images), torch.tensor(test_labels)
    )
    blurred_loader = torch.utils.data.DataLoader(blurred_dataset, batch_size=32)
    blurred_accuracy = evaluate_accuracy(model, blurred_loader, device)
    print(f"Test Accuracy with Blur: {blurred_accuracy * 100:.2f}%")

    # 对测试数据翻转并评估
    flipped_images = add_flip(test_images.numpy())
    flipped_dataset = torch.utils.data.TensorDataset(
        torch.tensor(flipped_images), torch.tensor(test_labels)
    )
    flipped_loader = torch.utils.data.DataLoader(flipped_dataset, batch_size=32)
    flipped_accuracy = evaluate_accuracy(model, flipped_loader, device)
    print(f"Test Accuracy with Flips: {flipped_accuracy * 100:.2f}%")

    # 确保准确性达到最低阈值
    assert original_accuracy > 0.7, "Original accuracy is below the expected threshold!"
    assert noisy_accuracy > 0.6, "Accuracy with noise is below the expected threshold!"
    assert blurred_accuracy > 0.4, "Accuracy with blur is below the expected threshold!"
    assert flipped_accuracy > 0.35, "Accuracy with flips is below the expected threshold!"

    # 创建柱状图来展示不同条件下的准确率
    conditions = ['Original', 'Noise', 'Blur', 'Flip']
    accuracies = [original_accuracy, noisy_accuracy, blurred_accuracy, flipped_accuracy]

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    plt.bar(conditions, accuracies, color=['blue', 'orange', 'green', 'red'], alpha=0.7)

    # 添加标题和坐标轴标签
    plt.title('Model Accuracy under Different Variations')
    plt.xlabel('Conditions')
    plt.ylabel('Accuracy')

    # 显示每个柱子顶部的准确率数值
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v * 100:.2f}%", ha='center', va='bottom')

    # 展示或保存图表
    plt.tight_layout()
    plt.savefig('chart/accuracy_accuracy.png')  # 如果你想要保存图表到文件，请取消注释此行
