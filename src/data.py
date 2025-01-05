<<<<<<< HEAD
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader  # 确保导入正确的 DataLoader
#
# def load_cifar10(batch_size=32):
#     transform = transforms.Compose([transforms.ToTensor()])
#     test_dataset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, transform=transform, download=True
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)  # 确保引用无误
#     return test_loader
# data.py
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader  # 确保导入正确的 DataLoader

def load_cifar10(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    # 加载训练数据
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision import datasets


# 数据集划分
def load_cifar10_fairness(batch_size=32):
    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    x_train, y_train = train_dataset.data, train_dataset.targets
    x_test, y_test = test_dataset.data, test_dataset.targets

    # 确保划分是随机的
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    train_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    val_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)

=======
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader  # 确保导入正确的 DataLoader
#
# def load_cifar10(batch_size=32):
#     transform = transforms.Compose([transforms.ToTensor()])
#     test_dataset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, transform=transform, download=True
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)  # 确保引用无误
#     return test_loader
# data.py
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader  # 确保导入正确的 DataLoader

def load_cifar10(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    # 加载训练数据
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision import datasets


# 数据集划分
def load_cifar10_fairness(batch_size=32):
    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    x_train, y_train = train_dataset.data, train_dataset.targets
    x_test, y_test = test_dataset.data, test_dataset.targets

    # 确保划分是随机的
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    train_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    val_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)

>>>>>>> f2afddc31a520a45b1ff325882be0b0ccd3500a1
    return (train_loader, val_loader), test_loader