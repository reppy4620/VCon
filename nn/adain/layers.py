import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .functions import tanhexp


class EncoderLayer(nn.Module):
    def __init__(self, c, ks):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(c, c, ks, 1, padding=(ks-1)//2, padding_mode='reflect'),
            nn.InstanceNorm1d(c, affine=True),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c, c, ks, 1, padding=(ks-1)//2, padding_mode='reflect'),
            nn.InstanceNorm1d(c, affine=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(c, c, ks, 1, padding=(ks-1)//2, padding_mode='reflect'),
            nn.InstanceNorm1d(c, affine=True),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(c, c, ks, 2, padding=(ks-1)//2, padding_mode='reflect'),
            nn.InstanceNorm1d(c, affine=True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = F.relu(x + self.block2(out))
        out = self.block3(out)
        out = F.relu(F.avg_pool1d(x, kernel_size=2, ceil_mode=True) + self.block4(out))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, c, ks, speaker_emb_dim):
        super().__init__()

        self.affines = nn.ModuleList([
            nn.Linear(speaker_emb_dim, c*2) for _ in range(4)
        ])

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=c,
                out_channels=c if i < 3 else c*2,  # for pixel shuffle
                kernel_size=ks,
                stride=1,
                padding=(ks-1)//2,
                padding_mode='reflect'
            ) for i in range(4)
        ])

        self.norm = nn.InstanceNorm1d(c, affine=False)

    @staticmethod
    def pixel_shuffle_1d(x, scale_factor=2):
        b, c, in_t = x.size()
        c //= scale_factor
        out_t = in_t * scale_factor
        x = x.reshape(b, c, scale_factor, in_t).transpose(-1, -2).reshape(b, c, out_t)
        return x

    @staticmethod
    def artificial_in(x, cond):
        s = cond.size(-1) // 2
        m, s = cond[:, :s], cond[:, s:]
        x = x * s.unsqueeze(-1) + m.unsqueeze(-1)
        return x

    def forward(self, x, cond):
        out = self.artificial_in(self.norm(self.convs[0](x)), self.affines[0](cond))
        out = F.relu(out)
        out = self.artificial_in(self.norm(self.convs[1](out)), self.affines[1](cond))
        out = F.relu(x + out)
        out = self.artificial_in(self.norm(self.convs[2](out)), self.affines[2](cond))
        out = F.relu(out)
        out = self.norm(self.convs[3](out))
        out = self.pixel_shuffle_1d(out)
        out = self.artificial_in(out, self.affines[3](cond))
        out = F.relu(F.interpolate(x, scale_factor=2) + out)
        return out


class DisLayer(nn.Module):
    def __init__(self, c_in, c_out, ks):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv1d(c_in, c_out, ks, padding=(ks-1)//2, padding_mode='reflect'))
        self.conv2 = spectral_norm(nn.Conv1d(c_out, c_out, ks, 2, padding=(ks-1)//2, padding_mode='reflect'))

    def forward(self, x):
        x = tanhexp(self.conv1(x))
        x = tanhexp(self.conv2(x))
        return x
