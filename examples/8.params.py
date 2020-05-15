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
print(params)

params.epoch = 400
params.batch_size = 25
print(params)


from thexp import Params
class MyParams(Params):
    def __init__(self):
        super().__init__()
        self.batch_size = 50
        self.topk = (1,2,3,4)
        self.optim = dict(
            lr=0.009,
            moment=0.9
        )

params = MyParams()
print(params)


from thexp import Params
params = Params()
params.choice("dataset","cifar10","cifar100","svhn")
params.arange("thresh",5,0,20)
print(params)

for g in params.grid_search("thresh",range(0,20)):
    for g in g.grid_search("dataset",['cifar10','cifar100','svhn']):
        print(g.dataset,g.thresh)

