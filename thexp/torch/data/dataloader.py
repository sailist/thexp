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

from torch.utils.data.dataloader import DataLoader as _DataLoader


class DataLoader(_DataLoader):
    """
    用于可调整 batch_size 的DataLoader，其需求基于

    Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le, Don't Decay the Learning Rate, Increase the Batch Size
    https://arxiv.org/abs/1711.00489
    """

    def set_batch_size(self, batch_size):
        self.batch_sampler.batch_size = batch_size
