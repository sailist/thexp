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


from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.asgd import ASGD
from torch.optim.lbfgs import LBFGS
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.sgd import SGD
from torch.optim.sparse_adam import SparseAdam
from torch.optim import adam,adamw,adamax,adagrad,adadelta,asgd,sparse_adam,sgd,lr_scheduler,lbfgs,optimizer,rmsprop,rprop
