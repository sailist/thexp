from torch.utils.data import Dataset


class SubsetWithTransform(Dataset):
    """
    对应于 torch.utils.data.dataset.Subset，在创建的时候可以选择对数据添加 transform
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.trasform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):

        img, target = self.dataset[self.indices[idx]]
        if self.trasform is not None:
            img = self.trasform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)
