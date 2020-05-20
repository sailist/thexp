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

from thexp import Trainer,Params
import random

class myTrainer(Trainer):
    pass

trainer = myTrainer(Params())

for i in range(50):
    trainer.logger.info(i)

# for i in range(500):
#     for j in 'qwertyuiopasdfgh':
#         trainer.writer.add_scalar(j, random.random(), i)

for i in range(20):
    trainer.writer.add_scalar('test', random.random(), i)

trainer.writer.add_hparams({"a":0.1,'b':3},{"acc":0.5,'loss':1})
trainer.writer.add_hparams({"a":0.2,'b':5},{"acc":0.1,'loss':4})
trainer.writer.add_hparams({"a":0.3,'b':2},{"acc":0.3,'loss':3})
