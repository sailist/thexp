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

import random

import numpy as np
import torch


def fix_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

def get_state():
    return {
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch.cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "random": random.getstate(),
    }

def set_state(state_dict):
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["numpy"])
    torch.random.set_rng_state(state_dict["torch"])
    if torch.cuda.is_available():
        if "torch.cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["torch.cuda"])
        else:
            import warnings
            warnings.warn("Don't have torch.cuda random state")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True



