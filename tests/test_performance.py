from src.data import load_cifar10
from src.model import load_resnet, train_model, load_model, save_model
from src.test_utils import evaluate_performance
import torch

from src.transforms import add_noise, add_blur, add_rotation


def test_model_performance():
    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 加载并训练模型
    model = load_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = train_model(model, train_loader, epochs=5, lr=0.001, device=device)
    model_path = "./data/resnet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)

    else:
        print("Training model...")
        model = train_model(model, train_loader, epochs=3, lr=0.001, device=device)
        save_model(model, path=model_path)  # 保存模型
        print(f"Model saved to {model_path}.")


    # 测试性能
    avg_inference_time = evaluate_performance(model, test_loader, device)
    print(f"Average Inference Time per Sample: {avg_inference_time:.6f} seconds")

    # 对测试数据添加噪声并评估
    test_images, test_labels = next(iter(test_loader))
    noisy_images = add_noise(test_images.numpy(), noise_level=0.1)
    noisy_dataset = torch.utils.data.TensorDataset(
        torch.tensor(noisy_images), torch.tensor(test_labels)
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)
    noisy_time = evaluate_performance(model, noisy_loader, device)
    print(f"Average Inference Time with Noise: {noisy_time:.6f} seconds")

    # 对测试数据添加模糊并评估
    blurred_images = add_blur(test_images.numpy(), kernel_size=5)
    blurred_dataset = torch.utils.data.TensorDataset(
        torch.tensor(blurred_images), torch.tensor(test_labels)
    )
    blurred_loader = torch.utils.data.DataLoader(blurred_dataset, batch_size=32)
    blurred_time = evaluate_performance(model, blurred_loader, device)
    print(f"Average Inference Time with Blur: {blurred_time:.6f} seconds")

    # 对测试数据翻转并评估
    flipped_images = add_rotation(test_images.numpy())
    flipped_dataset = torch.utils.data.TensorDataset(
        torch.tensor(flipped_images), torch.tensor(test_labels)
    )
    flipped_loader = torch.utils.data.DataLoader(flipped_dataset, batch_size=32)
    flipped_time = evaluate_performance(model, flipped_loader, device)
    print(f"Average Inference Time with Flips: {flipped_time:.6f} seconds")

    # 在你的 test_model_performance 函数中调用此函数
    plot_inference_times(avg_inference_time, noisy_time, blurred_time, flipped_time)

    # 确保性能在合理范围内
    assert avg_inference_time < 0.01, "Inference time is too high!"
    assert noisy_time < 0.02, "Inference time with noise is too high!"
    assert blurred_time < 0.02, "Inference time with blur is too high!"
    assert flipped_time < 0.02, "Inference time with flips is too high!"


import os
import matplotlib.pyplot as plt

def plot_inference_times(avg_inference_time, noisy_time, blurred_time, flipped_time):
    # 数据
    labels = ['Original', 'Noisy', 'Blurred', 'Flipped']
    times = [avg_inference_time, noisy_time, blurred_time, flipped_time]

    # 创建图表
    plt.figure(figsize=(8, 6))
    plt.bar(labels, times, color=['blue', 'orange', 'green', 'red'])

    # 添加标题和标签
    plt.title('Inference Time Comparison', fontsize=14)
    plt.xlabel('Test Type', fontsize=12)
    plt.ylabel('Inference Time (seconds)', fontsize=12)

    # 显示数值
    for i, time in enumerate(times):
        plt.text(i, time + 0.001, f'{time:.6f}', ha='center', fontsize=10)

    # 保存图表
    output_path = './inference_times_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

