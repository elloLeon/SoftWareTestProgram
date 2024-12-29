import pytest
from src.model import load_resnet
from src.data import load_cifar10


"""
    加载 ResNet 模型和 CIFAR-10 数据集，并提供测试数据。
"""

@pytest.fixture
def model():
    return load_resnet()

@pytest.fixture
def data():
    test_loader = load_cifar10()
    test_images, test_labels = next(iter(test_loader))
    return test_images.numpy(), test_labels.numpy()
