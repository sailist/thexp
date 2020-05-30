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
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
import time
print(__VERSION__)

import random
# from thexp.utils import random as rnd
import torch.nn as nn
import torch
from torch.optim import SGD


from thexp import RndManager
rnd = RndManager()

for i in range(2):
    rnd.mark("train")
    # rnd.fix_seed(1)
    data = torch.rand(5, 2)
    y = torch.tensor([0, 0, 0, 0, 0])
    model = nn.Linear(2, 2)

    sgd = SGD(model.parameters(), lr=0.01)
    logits = model(data)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    sgd.step()
    sgd.zero_grad()

    print(list(model.parameters()))

    rnd.shuffle()
    print(random.random())
