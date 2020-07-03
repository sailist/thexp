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
from typing import Union, List, Dict, Tuple

from thexp.utils.lazy import torch


def to_device(batch: Union[List, Dict, Tuple, torch().Tensor], device: torch().device):
    if isinstance(batch, (list, tuple)):
        return [to_device(ele, device) for ele in batch]
    elif isinstance(batch, dict):
        return {k: to_device(ele, device) for k, ele in batch.items()}
    elif isinstance(batch, torch().Tensor):
        return batch.to(device)
