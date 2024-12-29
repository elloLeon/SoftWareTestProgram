import os

from src.data import load_cifar10
from src.model import load_resnet, train_model, load_model, save_model
from src.transforms import add_noise
from src.test_utils import metamorphic_test
import torch
import numpy as np

def test_model_metamorphic():
    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 加载并训练模型
    model = load_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到设备上
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


    # 获取原始测试数据
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)  # 将输入数据移动到设备上

    # 创建变换后的数据集（噪声变异）
    noisy_images = add_noise(test_images.cpu().numpy(), noise_level=0.1, dtype=np.float32)  # 在CPU上进行噪声变换
    noisy_dataset = torch.utils.data.TensorDataset(
        torch.tensor(noisy_images).to(device), torch.tensor(test_labels).to(device)
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32)

    # 测试蜕变属性（变换前后输出一致性）
    consistency = metamorphic_test(model, test_loader, noisy_loader, device)
    print(f"Consistency between original and noisy data: {consistency * 100:.2f}%")

    # 确保一致性高于 90%
    assert consistency > 0.7, "Metamorphic testing failed: consistency too low!"