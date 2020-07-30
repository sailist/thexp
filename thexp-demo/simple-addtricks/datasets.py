"""

"""

from thexp import glob

root = glob['datasets']

def MNIST(mode="train", download=True):
    # 由于不同数据集区分训练或测试集的方式不同，因此建议数据集以统一的接口定义
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms
    weak = transforms.ToTensor()
    if mode == "train":
        return MNIST(root=root, train=True, download=download, transform=weak)
    else:
        return MNIST(root=root, train=False, transform=weak, download=download)


def FMNIST(mode="train", download=True):
    # 由于不同数据集区分训练或测试集的方式不同，因此建议数据集以统一的接口定义
    from torchvision.datasets.mnist import FashionMNIST
    from torchvision import transforms
    weak = transforms.ToTensor()
    if mode == "train":
        return FashionMNIST(root=root, train=True, download=download, transform=weak)
    else:
        return FashionMNIST(root=root, train=False, transform=weak, download=download)
