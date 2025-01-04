import os

from src.data import load_cifar10, load_cifar10_fairness
from src.model import load_resnet, train_model, load_model, save_model
from src.test_utils import evaluate_fairness
import torch
from collections import Counter

def test_model_fairness():
    # 统计类别分布
    _, test_loader = load_cifar10_fairness(batch_size=32)
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())

    label_counts = Counter(all_labels)
    print("Label distribution in test set:", label_counts)

    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size=32)

    # 加载并训练模型
    model = load_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./data/resnet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model, path=model_path, device=device)

    else:
        print("Training model...")
        model = train_model(model, train_loader, epochs=3, lr=0.001, device=device)
        save_model(model, path=model_path)  # 保存模型
        print(f"Model saved to {model_path}.")

    # 模拟敏感特征标签（例如，类别 0 和类别 1 作为敏感特征）
    sensitive_feature_indices = [0, 1]

    # 测试公平性
    fairness_reports = evaluate_fairness(model, test_loader, device, sensitive_feature_indices)

    # 输出公平性报告
    for category, report in fairness_reports.items():
        print(f"Fairness report for sensitive category {category}:")
        print(report)

    # 确保模型对每个类别的 F1 分数都高于 0.5
    for category, report in fairness_reports.items():
        assert report["macro avg"]["f1-score"] > 0.5, f"Fairness issue detected for category {category}!"
