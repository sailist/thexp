"""
_licence_content_
"""
from torchvision import datasets, transforms


def _mnist(train, transform):
    if train:
        return datasets.MNIST(root=args.dataset_path,
                              train=True,
                              transform=transform,
                              download=True)
    else:
        return datasets.MNIST(root=args.dataset_path,
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

def _cifar10(train, transform=None):
    if train:
        dataset = datasets.CIFAR10(root=args.dataset_path,
                                   train=True,
                                   transform=transform,
                                   download=True)
        return dataset
    else:
        return datasets.CIFAR10(root=args.dataset_path,
                                train=False,
                                transform=transforms.ToTensor())

_dataset_map = {
    "mnist": _mnist,
    "cifar10": _cifar10,
}

_n_classes_map = {
    "mnist": 10,
    "cifar10": 10,
}

def get_dataset(dataset="mnist", train=True, transform=None):
    if transform is None:
        transform = transforms.ToTensor()

    dataset = _dataset_map[dataset](train, transform=transform)
    n_classes = _n_classes_map[dataset]
    return dataset,n_classes