from src.data import load_cifar10
from src.model import load_resnet, train_model
from src.transforms import add_noise, add_blur, add_flip, add_rotation
from src.test_utils import evaluate_model  # 假设把 `evaluate_model` 放在 `src/test_utils.py`
import torch
import numpy as np
import matplotlib.pyplot as plt
import os  # 新增：用于检查文件路径
from src.model import save_model, load_model  # 新增：调用保存与加载模型的方法

def test_robustness_with_noise():
    # 加载数据
    model_path = "./data/resnet_model.pth"
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 初始化模型
    model = load_resnet()

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model = train_model(model, train_loader, epochs=5, lr=0.001, device=device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)

    else:
        print("Training model...")
        model = train_model(model, train_loader, epochs=5, lr=0.001, device=device)
        save_model(model, path=model_path)  # 保存模型
        print(f"Model saved to {model_path}.")
    # 原始数据准确率



    original_accuracy = evaluate_model(model, test_loader, device)

    # 获取测试图像和标签
    test_images, test_labels = next(iter(test_loader))

    # 添加噪声并计算准确率
    noisy_images = add_noise(test_images.numpy(), noise_level=0.1, dtype=np.float32)


# 1    至下一个"# 1"为添加多种变异的修改, 原来只有添加噪点一种,其实现在本部分下面
    """
    添加噪点 高斯模糊 水平翻转 旋转 四种变异处理
    """
    test_images_np = test_images.numpy()

    blurred_images = add_blur(test_images_np, kernel_size=5, method="gaussian")
    # 添加水平翻转
    flipped_images = add_flip(test_images_np, flip_code=1)
    # 添加旋转
    rotated_images = add_rotation(test_images_np, angle=15)
    # 创建变异后的 DataLoader
    def create_dataloader(transformed_images, labels):
        transformed_dataset = torch.utils.data.TensorDataset(
            torch.tensor(transformed_images, dtype=torch.float32),
            torch.tensor(labels)
        )
        return torch.utils.data.DataLoader(transformed_dataset, batch_size=32)

    loaders = {
        "Noise": create_dataloader(noisy_images, test_labels),
        "Blur": create_dataloader(blurred_images, test_labels),
        "Flip": create_dataloader(flipped_images, test_labels),
        "Rotate": create_dataloader(rotated_images, test_labels),
    }

    # 测试每种变异后的准确率
    accuracies = {"Original": original_accuracy}
    for name, loader in loaders.items():
        accuracy = evaluate_model(model, loader, device)
        accuracies[name] = accuracy
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    # 创建 chart 文件夹
    chart_folder = "./chart"
    os.makedirs(chart_folder, exist_ok=True)

    # 可视化原始图像与噪声图像
    visualize_save_path = os.path.join(chart_folder, "noise_vs_original.png")
    visualize_images(
        original_images=test_images[:5],
        noisy_images=noisy_images[:5],
        blurred_images=blurred_images[:5],
        flipped_images=flipped_images[:5],
        rotated_images=rotated_images[:5],
        labels=test_labels[:5],
        save_path=visualize_save_path
    )
    print(f"Visualization saved at {visualize_save_path}")

    # 绘制准确率对比图
    accuracy_save_path = os.path.join(chart_folder, "accuracy_comparison.png")
    plot_accuracy_drop(
        accuracies=accuracies,
        save_path=accuracy_save_path
    )
    print(f"Accuracy comparison saved at {accuracy_save_path}")

    # 返回测试结果
    return accuracies


# 1

def visualize_images(original_images, noisy_images, blurred_images, flipped_images, rotated_images, labels, save_path):
    """
    显示并保存原始图像与噪声图像的对比。
    """
    fig, axes = plt.subplots(5, len(original_images), figsize=(15, 5))
    for i in range(len(original_images)):
        # 将 PyTorch tensor 转换为 NumPy 数组，并转置为 (H, W, C) 以适应 plt.imshow
        original_image_np = original_images[i].numpy().transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
        noisy_image_np = noisy_images[i].transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
        blurred_image_np = blurred_images[i].transpose(1, 2, 0)
        flipped_image_np = flipped_images[i].transpose(1, 2, 0)
        rotated_image_np = rotated_images[i].transpose(1, 2, 0)

        # 归一化图像数据到 [0, 1] 范围内
        original_image_normalized = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        noisy_image_normalized = (noisy_image_np - noisy_image_np.min()) / (noisy_image_np.max() - noisy_image_np.min())
        blurred_image_normalized = (blurred_image_np - blurred_image_np.min()) / (blurred_image_np.max() - blurred_image_np.min())
        flipped_image_normalized = (flipped_image_np - flipped_image_np.min()) / (flipped_image_np.max() - flipped_image_np.min())
        rotated_image_normalized = (rotated_image_np - rotated_image_np.min()) / (rotated_image_np.max() - rotated_image_np.min())

        # 显示原始图像
        axes[0, i].imshow(original_image_normalized)
        axes[0, i].set_title(f"Original: {labels[i].item()}")
        axes[0, i].axis("off")

        # 显示模糊图像
        axes[1, i].imshow(blurred_image_normalized)
        axes[1, i].set_title(f"Blurred: {labels[i].item()}")
        axes[1, i].axis("off")

        # 显示水平翻转图像
        axes[2, i].imshow(flipped_image_normalized)
        axes[2, i].set_title(f"Flipped: {labels[i].item()}")
        axes[2, i].axis("off")

        # 显示旋转图像
        axes[3, i].imshow(rotated_image_normalized)
        axes[3, i].set_title(f"Rotated: {labels[i].item()}")
        axes[3, i].axis("off")

        # 显示噪声图像
        axes[4, i].imshow(noisy_image_normalized)
        axes[4, i].set_title("Noisy")
        axes[4, i].axis("off")


    plt.tight_layout()
    plt.savefig(save_path)  # 保存到文件
    plt.close()  # 关闭绘图窗口


def plot_accuracy_drop(accuracies, save_path):
    """
    绘制并保存所有变异类型的数据准确率对比图。
    参数：
        accuracies (dict): 一个包含变异类型及其对应准确率的字典，例如：
                          {"Original": 0.85, "Noise": 0.75, "Blur": 0.70, "Flip": 0.78, "Rotate": 0.72}
        save_path (str): 保存图片的路径。
    """
    labels = list(accuracies.keys())
    values = list(accuracies.values())

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel('Scenario')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Different Transformations')
    plt.ylim(0, 1)  # 确保显示范围在 [0, 1]

    # 在柱子顶部显示准确率数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)  # 保存到文件
    plt.close()  # 关闭绘图窗口
