import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return x / torch.norm(x, dim=1, keepdim=True)
