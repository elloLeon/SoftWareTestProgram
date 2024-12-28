from src.data import load_cifar10
from src.model import load_resnet, train_model
from src.transforms import add_noise
from src.test_utils import evaluate_model  # 假设把 `evaluate_model` 放在 `src/test_utils.py`
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_robustness_with_noise():
    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 初始化模型
    model = load_resnet()

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(model, train_loader, epochs=5, lr=0.001, device=device)

    # 原始数据准确率
    original_accuracy = evaluate_model(model, test_loader, device)

    # 获取测试图像和标签
    test_images, test_labels = next(iter(test_loader))

    # 添加噪声并计算准确率
    noisy_images = add_noise(test_images.numpy(), noise_level=0.1, dtype=np.float32)

    # 创建 DataLoader 包装 noisy_images
    noisy_dataset = torch.utils.data.TensorDataset(
        torch.tensor(noisy_images), torch.tensor(test_labels)
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)

    # 噪声数据准确率
    noise_accuracy = evaluate_model(model, noisy_loader, device)

    # print("原始图像数据：")
    # print(test_images[:5])
    # print("噪声图像数据：")
    # print(noisy_images[:5])
    # print("标签：")
    # print(test_labels[:5])

    # # 可视化变异前后的数据
    # visualize_images(test_images[:5], noisy_images[:5], test_labels[:5], save_path="data_visualization.png")
    #
    # # 可视化测试对比结果
    # plot_accuracy_drop(original_accuracy, noise_accuracy, save_path="accuracy_comparison.png")

    # 打印测试结果
    print(f"Original Accuracy: {original_accuracy * 100:.2f}%")
    print(f"Noise Accuracy: {noise_accuracy * 100:.2f}%")

    # 返回测试准确率（可选）
    return original_accuracy, noise_accuracy
# def test_robustness_with_noise(model, data):
#     print("进入 test_robustness_with_noise 函数")
#     test_images, test_labels = data
#
#     # 原始数据准确率
#     original_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(torch.tensor(test_images).float(), torch.tensor(test_labels)),
#         batch_size=32
#     )
#     original_accuracy = evaluate_model(model.float(), original_loader)
#
#     # 将生成的噪声图像转换为 Float 类型
#     noisy_images = add_noise(test_images, noise_level=0.1, dtype=np.float32)
#
#     # 创建 DataLoader 包装 noisy_images
#     noisy_dataset = torch.utils.data.TensorDataset(
#         torch.tensor(noisy_images), torch.tensor(test_labels)
#     )
#     noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)
#
#     # 噪声数据准确率
#     noise_accuracy = evaluate_model(model, noisy_loader)
#     print("原始图像数据：")
#     print(test_images[:5])
#     print("噪声图像数据：")
#     print(noisy_images[:5])
#     print("标签：")
#     print(test_labels[:5])
#
#     # 可视化变异前后的数据
#     visualize_images(test_images[:5], noisy_images[:5], test_labels[:5], save_path="data_visualization.png")
#
#     # 可视化测试对比结果
#     plot_accuracy_drop(original_accuracy, noise_accuracy, save_path="accuracy_comparison.png")
#
#
#     # 打印测试结果
#     print(f"Original Accuracy: {original_accuracy * 100:.2f}%")
#     print(f"Noise Accuracy: {noise_accuracy * 100:.2f}%")
#
#     # 返回测试准确率（可选）
#     return original_accuracy, noise_accuracy
#
#
# # def visualize_images(original_images, noisy_images, labels, save_path):
# #     """
# #     显示并保存原始图像与噪声图像的对比。
# #     """
# #     fig, axes = plt.subplots(2, len(original_images), figsize=(15, 5))
# #     for i in range(len(original_images)):
# #         # 显示原始图像
# #         axes[0, i].imshow(original_images[i].transpose(1, 2, 0))  # 转换维度为 HWC
# #         axes[0, i].set_title(f"Original: {labels[i]}")
# #         axes[0, i].axis("off")
# #
# #         # 显示噪声图像
# #         axes[1, i].imshow(noisy_images[i].transpose(1, 2, 0))  # 转换维度为 HWC
# #         axes[1, i].set_title("Noisy")
# #         axes[1, i].axis("off")
# #
# #     plt.tight_layout()
# #     plt.savefig(save_path)  # 保存到文件
# #     plt.close()  # 关闭绘图窗口

# 这个地方可视化貌似出了点问题，需要修改，前面注释了

def visualize_images(original_images, noisy_images, labels, save_path):
    """
    显示并保存原始图像与噪声图像的对比。
    """
    fig, axes = plt.subplots(2, len(original_images), figsize=(15, 5))
    for i in range(len(original_images)):
        # 将 PyTorch tensor 转换为 NumPy 数组，并转置为 (H, W, C) 以适应 plt.imshow
        original_image_np = original_images[i].numpy().transpose(1, 2, 0)
        noisy_image_np = noisy_images[i].transpose(1, 2, 0)

        # 归一化图像数据到 [0, 1] 范围内
        original_image_normalized = (original_image_np.transpose(1, 2, 0) - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        noisy_image_normalized = (noisy_image_np.transpose(1, 2, 0) - noisy_image_np.min()) / (noisy_image_np.max() - noisy_image_np.min())

        # 显示原始图像
        axes[0, i].imshow(original_image_normalized)
        axes[0, i].set_title(f"Original: {labels[i]}")
        axes[0, i].axis("off")

        # 显示噪声图像
        axes[1, i].imshow(noisy_image_normalized)
        axes[1, i].set_title("Noisy")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)  # 保存到文件
    plt.close()  # 关闭绘图窗口

def plot_accuracy_drop(original_acc, noise_acc, save_path):
    """
    绘制并保存变异前后数据的准确率对比图。
    """
    labels = ['Original', 'Noise']
    accuracies = [original_acc, noise_acc]
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.xlabel('Scenario')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)  # 确保显示范围在 [0, 1]
    plt.savefig(save_path)  # 保存到文件
    plt.close()  # 关闭绘图窗口
