"""
_licence_content_
"""
import torch.nn as nn
from torch.nn.utils import weight_norm


class MLP(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.fc = weight_norm(nn.Linear(28*28, n_classes))

    def forward(self, x):
        out = x
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
