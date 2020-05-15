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

from collections import OrderedDict
from itertools import cycle, chain

from torch.utils.data import DataLoader
from ..utils.date.dataloader import DataLoader as thDataLoader
import torch




class DataBundler:
    """
    当一个模型需要训练多个数据集的时候，通过DataBundler类和附属的装饰器灵活的提供数据集::

        bundler = DataBundler() \
                .cycle(cifar_dataloader, "cifar") \
                .add(svhn_dataloader, "svhn").chain_iter()

        for (imgs,labels) in bundler:
            ...

    效果等价于::

        for (imgs,chains) in chain(cifar_dataloader,svhn_dataloader):
            ...

    另外还支持
    """

    class DataBundler(OrderedDict):
        pass

    def __init__(self):
        self.dataloaders = DataBundler.DataBundler()
        self.iter_mode = "chain"

    def set_batch_size(self,batch_size):
        for _,(loader,_) in self.dataloaders.items():
            if isinstance(loader,thDataLoader):
                loader.set_batch_size(batch_size)
            elif isinstance(loader,DataLoader):
                loader.batch_sampler.batch_size = batch_size

    def _append(self, loader, func, name):
        assert isinstance(loader, (DataLoader,DataBundler))
        if name is None:
            unname = "unnamed"
            i = 0
            name = "{}_{}".format(unname, i)
            while name in self.dataloaders:
                i += 1
                name = "{}_{}".format(unname, i)
        else:
            assert name not in self.dataloaders, "{} also defined in bundler".format(name)

        self.dataloaders[name] = (loader, func)

    def __len__(self):
        if self.iter_mode == 'zip':
            return max(*self.len_list())
        elif self.iter_mode == "chain":
            return sum(self.len_list())

    def len_list(self):
        """
        按照添加的顺序返回各个dataloader的长度（batch级别）
        :return:
        """
        return [len(loader) for _, (loader, _) in self.dataloaders.items()]

    def len_dict(self):
        """
        返回每个loader的 name:len 字典
        :return: an OrderedDict
        """
        res = DataBundler.DataBundler()
        for name, (loader, func) in self.dataloaders.items():
            res[name] = len(loader)
        return res

    def cycle(self, loader: DataLoader, name=None):
        """一般在zip中保证数据量少的数据集不会成为拖累"""
        self._append(loader, cycle, name)
        return self

    def add(self, loader, name=None):
        self._append(loader, lambda x: x, name)
        return self

    def _func_loader(self):
        def filter(loader):
            if isinstance(loader, DataBundler):
                return iter(loader)
            return loader

        return [func(filter(loader)) for name, (loader, func) in self.dataloaders.items()]

    def __getitem__(self, item):
        return self.dataloaders[item][0]

    def __iter__(self):
        loaders = self._func_loader()
        if len(loaders) == 1:
            return iter(loaders[0])

        if self.iter_mode == "zip":
            return zip(*loaders)
        elif self.iter_mode == "chain":
            return chain(*loaders)

        assert False

    def choice_batch(self)->tuple:
        return next(iter(self))

    def choice_sample(self)->tuple:
        xs,ys = next(iter(self)) # type:(torch.Tensor,torch.Tensor)
        return (xs[0],ys[0])

    def zip_mode(self):
        """切换为zip方式提供所有添加的数据集"""
        self.iter_mode = "zip"
        return self

    def chain_mode(self):
        """
        切换为chain方式提供所有添加的数据集
            注意，如果以cycle方法添加了某个数据集，那么该迭代将永远不会停止
        :return:
        """
        self.iter_mode = "chain"
        return self

    @staticmethod
    def create_zip_bundler(**kwargs):
        bundler = DataBundler()
        loaders = [(len(v),v,k) for k,v in kwargs.items()]
        loaders.sort(reverse=True)
        bundler.add(loaders[0][1],loaders[0][2])
        for (_,loader,name) in loaders[1:]:
            bundler.cycle(loader,name)
        return bundler

    @staticmethod
    def create_chain_bundler(*args):
        bundler = DataBundler()
        for loader in args:
            bundler.add(loader)
        return bundler


    def __repr__(self):
        from pprint import pformat
        return pformat(self.len_dict())