import torch
import torch.nn as nn


class TanhExp(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
