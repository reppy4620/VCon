import torch
import torch.nn.functional as F


def swish(x, beta=1):
    return x * torch.sigmoid(beta * x)


def mish(x):
    return x * torch.tanh(F.softplus(x))


def tanhexp(x):
    return x * torch.tanh(torch.exp(x))
