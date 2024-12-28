import pytest
from src.model import load_resnet
from src.data import load_cifar10

@pytest.fixture
def model():
    return load_resnet()

@pytest.fixture
def data():
    test_loader = load_cifar10()
    test_images, test_labels = next(iter(test_loader))
    return test_images.numpy(), test_labels.numpy()
