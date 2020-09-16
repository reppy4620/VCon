import torch


def tanhexp(x):
    return x * torch.tanh(torch.exp(x))
