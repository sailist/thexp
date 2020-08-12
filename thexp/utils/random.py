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
    fix_cuda()
    return get_state()


def fix_cuda():
    if torch.cuda.is_available():
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
    fix_cuda()