"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
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
