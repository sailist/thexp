from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from typing import Tuple


def walk_module(module: nn.Module) -> Tuple[nn.Module, str, nn.Parameter]:
    for name, submodule in module.named_children():
        for ssubmodule, subname, subparam in walk_module(submodule):
            yield ssubmodule, subname, subparam

    for pname, param in module.named_parameters(recurse=False):
        yield module, pname, param


class ParamGrouper:
    def __init__(self, module: nn.Module):
        self.module = module

    def params(self, with_norm=True):
        params = []
        for module, name, param in walk_module(self.module):
            if with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm)):
                params.append(param)
        return params

    def kernel_params(self, with_norm=True):
        params = []
        for module, name, param in walk_module(self.module):
            if 'weight' in name and (with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm))):
                params.append(param)
        return params

    def bias_params(self, with_norm=True):
        params = []
        for module, name, param in walk_module(self.module):
            if 'bias' in name and (with_norm or not isinstance(module, (_BatchNorm, nn.LayerNorm))):
                params.append(param)
        return params

    def batchnorm_params(self):
        return [param for module, name, param in walk_module(self.module) if isinstance(module, _BatchNorm)]

    def layernorm_params(self):
        return [param for module, name, param in walk_module(self.module) if isinstance(module, nn.LayerNorm)]

    def norm_params(self):
        return [param for module, name, param in walk_module(self.module) if
                isinstance(module, (_BatchNorm, nn.LayerNorm))]

    def create_param_group(self, params, **kwargs):
        kwargs['params'] = params
        return kwargs
