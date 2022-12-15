import torch
from torch import nn

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2
