import torch.nn as nn

from .activations import Activation


def Conv1d(c_in: int, c_out: int, k: int, s: int = 1):
    p = (k - 1) // 2
    conv = nn.Conv1d(c_in, c_out, k, s, p)
    nn.init.kaiming_normal_(conv.weight)
    return conv


class InstanceNorm1d(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean, std = self.mean_std(x, self.eps)
        x = (x - mean) / std
        return x, (mean, std)

    @staticmethod
    def mean_std(x, eps=1e-5):
        B, C = x.shape[:2]

        x = x.view(B, C, -1)
        mean = x.mean(-1).view(B, C, 1)
        std = (x.std(-1) + eps).view(B, C, 1)

        return mean, std


class EncoderLayer(nn.Module):

    def __init__(self, c_in, c_h, act_name):
        super().__init__()
        self.layer = nn.Sequential(
            Conv1d(c_in, c_h, 3),
            nn.BatchNorm1d(c_h),
            Activation(act_name),
            Conv1d(c_h, c_in, 3)
        )

    def forward(self, x):
        return self.layer(x)


class DecoderLayer(nn.Module):
    def __init__(self, c_in, c_h, act_name):
        super().__init__()
        self.layer1 = nn.Sequential(
            Conv1d(c_in, c_h, 3),
            nn.BatchNorm1d(c_h),
            Activation(act_name),
            Conv1d(c_h, c_in, 3)
        )
        self.layer2 = nn.Sequential(
            Conv1d(c_in, c_h, 3),
            nn.BatchNorm1d(c_h),
            Activation(act_name),
            Conv1d(c_h, c_in, 3)
        )

    def forward(self, x):
        h = self.layer1(x)
        h = h + self.layer2(x)
        return x + h
