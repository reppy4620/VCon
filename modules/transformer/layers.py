import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import AttributeDict


def Conv1d(c_in: int, c_out: int, k: int, s: int = 1, d: int = 1):
    p = (k - 1) // 2
    conv = nn.Conv1d(c_in, c_out, k, s, p, dilation=d)
    nn.init.kaiming_normal_(conv.weight)
    return conv


class ResidualConv(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.layer1 = nn.Sequential(
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel)
        )

    def forward(self, src: Tensor) -> Tensor:
        x = self.layer1(src)
        x = src + self.layer2(x)
        x = F.gelu(x)
        return x


class ConvExtractor(nn.Module):
    def __init__(self, in_c, middle_c, out_c, n_layer):
        super().__init__()

        self.in_conv = nn.Sequential(
            Conv1d(in_c, middle_c, 9),
            nn.BatchNorm1d(middle_c),
            nn.GELU()
        )

        self.conv_layers = nn.Sequential(*[ResidualConv(middle_c) for _ in range(n_layer-2)])

        self.out_conv = nn.Sequential(
            Conv1d(middle_c, out_c, 5),
            nn.BatchNorm1d(out_c),
            nn.GELU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.conv_layers(x)
        x = self.out_conv(x)
        return x


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.empty(dim, n_embed).normal_()
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, params: AttributeDict, is_ffn: bool = True):
        super().__init__()

        channel = params.model.channel

        self.attn = nn.MultiheadAttention(channel, params.model.n_head, params.model.dropout)
        self.attn_norm = nn.LayerNorm(channel)
        self.attn_dropout = nn.Dropout(params.model.dropout)

        if is_ffn:
            self.ffn = nn.Sequential(
                Conv1d(channel, channel*2, 1),
                nn.GELU(),
                Conv1d(channel*2, channel, 1)
            )
            self.ffn_norm = nn.LayerNorm(channel)
            self.ffn_dropout = nn.Dropout(params.model.dropout)
        self.is_ffn = is_ffn

        self.attn_map = None

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        x, self.attn_map = self.attn(query, key, value)
        x = query + self.attn_dropout(x)
        x = self.attn_norm(x)
        return x

    def _ffn(self, src: Tensor) -> Tensor:
        # (L, B, C) => (B, C, L)
        x = src.permute(1, 2, 0)
        x = self.ffn(x)
        # (B, C, L) => (L, B, C)
        x = x.permute(2, 0, 1)

        x = src + self.ffn_dropout(x)
        x = self.ffn_norm(x)
        return x

    def forward(self, *args):
        raise NotImplementedError()


class SelfAttention(MultiHeadAttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        x = self._attn(x, x, x)
        if self.is_ffn:
            x = self._ffn(x)
        return x


class SourceTargetAttention(MultiHeadAttentionLayer):
    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self._attn(x, mem, mem)
        if self.is_ffn:
            x = self._ffn(x)
        return x
