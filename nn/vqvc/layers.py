import torch
import torch.nn as nn


class EncoderLayer(nn.Module):

    def __init__(self, in_channel, out_channel, middle_channel, k1, k2):
        super().__init__()

        self.extract_conv = nn.Sequential(
            nn.Conv1d(in_channel, middle_channel, k1, 1, k1//2),
            nn.BatchNorm1d(middle_channel),
            nn.ReLU()
        )

        self.rnn = RCBlock(middle_channel, k1, 1)

        self.down_conv = nn.Sequential(
            nn.Conv1d(middle_channel, out_channel, k2, 2, k2//2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.extract_conv(x)
        x = self.rnn(x)
        x = self.down_conv(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channel, out_channel, middle_channel, emb_channel, k1, k2):
        super().__init__()

        self.adjust_conv = nn.Sequential(
            nn.Conv1d(in_channel * 2, in_channel, k1-1, 1, (k1-1)//2, groups=2),
            nn.GroupNorm(2, in_channel),
            nn.ReLU()
        )

        self.extract_conv = nn.Sequential(
            nn.Conv1d(in_channel + emb_channel, middle_channel, k1-1, 1, (k1-1)//2),
            nn.BatchNorm1d(middle_channel),
            nn.ReLU()
        )

        self.refine_conv1 = nn.Sequential(
            nn.Conv1d(middle_channel, middle_channel, k1-1, 1, (k1-1)//2),
            nn.BatchNorm1d(middle_channel),
            nn.ReLU()
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(middle_channel, middle_channel, k1, 2, k1//2 - 1),
            nn.BatchNorm1d(middle_channel),
            nn.ReLU()
        )

        self.rnn = RCBlock(middle_channel, k1-1, 1)

        self.refine_conv2 = nn.Sequential(
            nn.Conv1d(middle_channel, out_channel, k2, 1, k2//2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, emb):
        # First, x is concatenated with q_after, so adjust channel dim.
        x = self.adjust_conv(x)
        emb = emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = self.extract_conv(torch.cat([x, emb], dim=1))
        x = x + self.refine_conv1(x)
        x = self.up_conv(x)
        x = self.rnn(x)
        x = self.refine_conv2(x)
        return x


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


class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation):
        super().__init__()
        self.rec = nn.GRU(feat_dim, feat_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size=ks,
                stride=1,
                padding=(ks - 1) * dilation // 2,
                dilation=dilation,
                groups=2
            ),
            nn.GroupNorm(2, feat_dim),
            nn.ReLU()
        )

        self.insert = Insert()

    def forward(self, x):
        r, _ = self.rec(x.transpose(1, 2))
        r = r.transpose(1, 2)
        c = self.conv(self.insert(r))
        return r+c


class Insert(nn.Module):
    def forward(self, x):
        B, C, L = x.size()
        x = x.view(B, 2, C//2, L).transpose(1, 2)
        x = x.contiguous().view(B, C, L)
        return x
