"""

"""
from typing import Union, List, Dict, Tuple

import torch


def to_device(batch: Union[List, Dict, Tuple, torch.Tensor], device: torch.device):
    if isinstance(batch, (list, tuple)):
        return [to_device(ele, device) for ele in batch]
    elif isinstance(batch, dict):
        return {k: to_device(ele, device) for k, ele in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
