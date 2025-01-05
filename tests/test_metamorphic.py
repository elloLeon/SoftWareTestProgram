import os
import matplotlib.pyplot as plt
from src.data import load_cifar10
from src.model import load_resnet, train_model, load_model, save_model
from src.transforms import add_noise
from src.test_utils import metamorphic_test
import torch
import numpy as np

def visualize_predictions_and_save(model, original_images, noisy_images, labels, device, save_dir="./chart"):
    """可视化原始和添加噪声后的图片及其预测结果，并生成柱状图保存图片"""
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 对原始图片进行预测
        original_preds = model(original_images).argmax(dim=1)
        # 对添加噪声后的图片进行预测
        noisy_preds = model(noisy_images).argmax(dim=1)

    # 选择前8张图片进行可视化
    num_images = 8
    fig, axes = plt.subplots(2, num_images, figsize=(20, 5))
    for i in range(num_images):
        # 显示原始图片
        axes[0, i].imshow(original_images[i].cpu().permute(1, 2, 0))
        axes[0, i].set_title(f"Label: {labels[i].item()}\nPred: {original_preds[i].item()}")
        axes[0, i].axis("off")

        # 显示添加噪声后的图片
        axes[1, i].imshow(noisy_images[i].cpu().permute(1, 2, 0))
        axes[1, i].set_title(f"Noisy Pred: {noisy_preds[i].item()}")
        axes[1, i].axis("off")

    # 保存图片到本地
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metamorphic_images.png"))
    plt.close(fig)  # 关闭当前图表，避免内存溢出

    # 生成准确度的柱状图
    original_accuracy = (original_preds == labels).sum().item() / len(labels)
    noisy_accuracy = (noisy_preds == labels).sum().item() / len(labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Original", "Noisy"], [original_accuracy, noisy_accuracy], color=["blue", "orange"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy on Original vs Noisy Images")
    plt.savefig(os.path.join(save_dir, "metamorphic.png"))
    plt.close(fig)

    print(f"Images and charts saved to {save_dir}")

def test_model_metamorphic():
    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 加载并训练模型
    model = load_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到设备上
    model_path = "./data/resnet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)
    else:
        print("Training model...")
        model = train_model(model, train_loader, epochs=3, lr=0.001, device=device)
        save_model(model, path=model_path)
        print(f"Model saved to {model_path}.")

    # 获取原始测试数据
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    # 创建变换后的数据集（噪声变异）
    noisy_images = add_noise(test_images.cpu().numpy(), noise_level=0.1, dtype=np.float32)
    noisy_dataset = torch.utils.data.TensorDataset(
        torch.tensor(noisy_images).to(device), torch.tensor(test_labels).to(device)
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)

    # 测试蜕变属性（变换前后输出一致性）
    consistency = metamorphic_test(model, test_loader, noisy_loader, device)
    print(f"Consistency between original and noisy data: {consistency * 100:.2f}%")
    #assert consistency > 0.7, "Metamorphic testing failed: consistency too low!"

    # 可视化预测结果并保存
    visualize_predictions_and_save(model, test_images, torch.tensor(noisy_images).to(device), test_labels, device)

if __name__ == "__main__":
    test_model_metamorphic()