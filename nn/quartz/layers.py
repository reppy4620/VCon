import torch
import torch.nn as nn
import torch.nn.functional as F


class QuartzLayer(nn.Module):

    def __init__(self, c_in, c_out, kernel, stride, pad, act=None):
        super().__init__()
        # depth-wise
        self.dw = nn.Conv1d(c_in, c_in, kernel, stride, pad, groups=c_in)
        # point-wise
        self.pw = nn.Conv1d(c_in, c_out, 1)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = act

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x, inplace=True)
        return x


class QuartzBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride, pad, n_block):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            QuartzLayer(
                c_in,
                c_out,
                kernel,
                stride,
                pad,
                act=F.relu
            )
        )
        for i in range(n_block - 2):
            self.layers.append(
                QuartzLayer(
                    c_out,
                    c_out,
                    kernel,
                    stride,
                    pad,
                    act=F.relu
                )
            )

        self.skip = nn.Sequential(
            # point-wise
            nn.Conv1d(c_in, c_out, 1),
            nn.BatchNorm1d(c_out)
        )

        self.out = QuartzLayer(
            c_out,
            c_out,
            kernel,
            stride,
            pad
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        skip_out = self.skip(x)
        return F.relu(out + skip_out)
