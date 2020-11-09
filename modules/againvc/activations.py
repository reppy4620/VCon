import torch
import torch.nn as nn
import torch.nn.functional as F


class VariantSigmoid(nn.Module):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * x))


class TanhExp(nn.Module):

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Activation(nn.Module):
    d = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'gelu': nn.GELU,
        'tanhexp': TanhExp,
        'mish': Mish,
        'v-sigmoid': VariantSigmoid
    }

    def __init__(self, name):
        super().__init__()
        self.act = self.d[name]()

    def forward(self, x):
        return self.act(x)
