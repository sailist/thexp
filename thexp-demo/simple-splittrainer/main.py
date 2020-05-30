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

sys.path.insert(0, r"E:\Python\iLearn\thexp")
from thexp import __VERSION__

print(__VERSION__)

# glob.add_value('datasets', '/home/share/yanghaozhe/pytorchdataset', glob.LEVEL.repository)

from config import params
from trainer.mnisttrainer import MNISTTrainer

# params
for p in params.grid_search("optim.lr", [0.001, 0.005, 0.0005]):
    for pp in p.grid_search("epoch", [10, 15, 20]):
        trainer = MNISTTrainer(pp)
        trainer.train()
