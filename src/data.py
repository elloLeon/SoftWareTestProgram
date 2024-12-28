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
