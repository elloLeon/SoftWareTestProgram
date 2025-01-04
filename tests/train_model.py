import os
import torch
from src.data import load_cifar10
from src.model import load_resnet, train_model, save_model, load_model

def train_or_load_model():
    """
    加载或训练模型。如果模型文件存在，则加载并继续训练一定轮数；否则，训练新模型。
    """
    train_loader, test_loader = load_cifar10(batch_size=32)
    model_path = "./data/resnet_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = load_resnet()

    if os.path.exists(model_path):
        # 加载已保存的模型
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)

        # 继续训练一定轮数
        print("Continuing training for additional epochs...")
        model = train_model(model, train_loader, epochs=2, lr=0.001, device=device)
    else:
        # 如果模型不存在，进行完整训练
        print("Training model from scratch...")
        model = train_model(model, train_loader, epochs=5, lr=0.001, device=device)

    # 保存模型
    save_model(model, path=model_path)
    print(f"Model saved to {model_path}.")

    return model

# 调用函数
model = train_or_load_model()
