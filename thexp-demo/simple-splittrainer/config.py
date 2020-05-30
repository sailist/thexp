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
from thexp import Params

params = Params()
params.device = 'cuda:1'
params.epoch = 5
params.batch_size = 128
params.topk = (1, 4)
params.from_args()
params.root = '/home/share/yanghaozhe/pytorchdataset'
params.dataloader = dict(shuffle=True, batch_size=32, drop_last=True)
params.optim = dict(lr=0.01, weight_decay=0.09, momentum=0.9)
params.choice('dataset', 'mnist', 'fmnist')
params.dataset = 'mnist'
params.bind('dataset', 'mnist', 'arch', 'simple')
params.bind('dataset', 'fmnist', 'arch', 'simple')
params.bind('dataset', 'cifar10', 'arch', 'cnn13')
params.ema = True
