import torch.nn as nn
from torch import Tensor

from typing import Optional, Tuple

from utils import AttributeDict


def Conv1d(c_in: int, c_out: int, k: int, s: int = 1, d: int = 1, padding_mode='zeros'):
    p = (k - 1) // 2
    conv = nn.Conv1d(c_in, c_out, k, s, p, dilation=d, padding_mode=padding_mode)
    nn.init.kaiming_normal_(conv.weight)
    return conv


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, params: AttributeDict, is_ffn: bool):
        super().__init__()

        channel = params.model.channel

        self.attn = nn.MultiheadAttention(channel, params.model.n_head, params.model.dropout)
        self.attn_norm = nn.LayerNorm(channel)
        self.attn_dropout = nn.Dropout(params.model.dropout)

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = nn.Sequential(
                Conv1d(channel, channel*2, 9),
                nn.GELU(),
                Conv1d(channel*2, channel, 1)
            )
            self.ffn_norm = nn.LayerNorm(channel)
            self.ffn_dropout = nn.Dropout(params.model.dropout)

    def _attn(self,
              query: Tensor,
              key: Tensor,
              value: Tensor,
              attn_mask: Optional[Tensor] = None,
              key_padding_mask: Optional[Tensor] = None
              ) -> Tuple[Tensor, Optional[Tensor]]:
        x, attn_map = self.attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.attn_dropout(x)
        x = self.attn_norm(x)
        return x, attn_map

    def _ffn(self, x: Tensor) -> Tensor:
        # (L, B, C) => (B, C, L)
        x = x.permute(1, 2, 0)
        x = self.ffn(x)
        # (B, C, L) => (L, B, C)
        x = x.permute(2, 0, 1)

        x = x + self.ffn_dropout(x)
        x = self.ffn_norm(x)
        return x

    def forward(self, *args):
        raise NotImplementedError()


class SelfAttention(MultiHeadAttentionLayer):
    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor]]:
        x, attn_map = self._attn(x, x, x, attn_mask, key_padding_mask)
        if self.is_ffn:
            x = self._ffn(x)
        return x, attn_map


class SourceTargetAttention(MultiHeadAttentionLayer):
    def forward(self,
                x: Tensor,
                mem: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor]]:
        x, attn_map = self._attn(x, mem, mem, attn_mask, key_padding_mask)
        if self.is_ffn:
            x = self._ffn(x)
        return x, attn_map
