import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(n_embed, dim, max_norm=1)

    def forward(self, x):
        embed = (self.embedding.weight.detach()).transpose(0, 1)

        embed = embed / (torch.norm(embed, dim=0))
        flatten = x.reshape(-1, self.dim).detach()

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1]).detach()
        quantize = self.embedding(embed_ind)
        diff = (quantize - x).pow(2).mean()
        quantize_1 = x + (quantize - x).detach()

        return (quantize + quantize_1) / 2, diff


class GBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, middle_dim, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(num_groups, middle_dim),
            nn.ReLU(),
            RCBlock(middle_dim, 3, dilation=1, num_groups=num_groups),
            nn.Conv1d(middle_dim, output_dim, 3, 1, 1, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        self.rec = nn.GRU(feat_dim, feat_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=ks,
            stride=1,
            padding=(ks-1)*dilation//2,
            padding_mode='reflect',
            dilation=dilation,
            groups=num_groups
        )
        self.gn = nn.GroupNorm(num_groups, feat_dim)

    def forward(self, x):
        r, _ = self.rec(x.transpose(1, 2))
        c = F.relu(self.gn(self.conv(r.transpose(1, 2))))
        return x+c
