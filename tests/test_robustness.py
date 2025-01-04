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
    for name, loader in loaders.items():
        accuracy = evaluate_model(model, loader, device)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")



    # 返回测试结果
    return {
        "Original": original_accuracy,
        "Noise": loaders["Noise"],
        "Blur": loaders["Blur"],
        "Flip": loaders["Flip"],
        "Rotate": loaders["Rotate"],}
# 1

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