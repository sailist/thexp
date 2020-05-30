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
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.fakedata import FakeData
from torchvision.transforms import ToTensor

from thexp import DataBundler

bundler = DataBundler()

sub = DataBundler()
sub.add(DataLoader(FakeData(transform=ToTensor()), batch_size=10)) \
    .add(DataLoader(FakeData(image_size=(3, 32, 32), transform=ToTensor()), batch_size=10)) \
    .zip_mode()

bundler.add(sub) \
    .add(DataLoader(FakeData(image_size=(3, 28, 28), transform=ToTensor()), batch_size=10)) \
    .zip_mode()

for ((i1, l1), (i2, l2)), (i3, l3) in bundler:
    print(i1.shape, l1.shape, i2.shape, l2.shape, i3.shape, l3.shape)

bundler = (
    DataBundler()
        .cycle(DataLoader(FakeData(size=10, image_size=(3, 28, 28), transform=ToTensor()), batch_size=10))
        .add(DataLoader(FakeData(size=1000, image_size=(3, 28, 28), transform=ToTensor()), batch_size=10))
        .zip_mode()
)

for i, ((i1, l1), (i2, l2)) in enumerate(bundler):
    print(i, i1.shape, l1.shape, i2.shape, l2.shape)
