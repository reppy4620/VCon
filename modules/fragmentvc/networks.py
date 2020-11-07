from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from utils import AttributeDict
from .layers import Conv1d, SelfAttention, SourceTargetAttention


class Extractor(nn.Module):
    def __init__(self, params: AttributeDict):
        super().__init__()

        self.self_attn = SelfAttention(params, is_ffn=False)
        self.st_attn = SourceTargetAttention(params, is_ffn=True)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_attn_mask: Optional[Tensor],
                tgt_attn_mask: Optional[Tensor],
                src_key_padding_mask: Optional[Tensor],
                tgt_key_padding_mask: Optional[Tensor]
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        src, self_attn_map = self.self_attn(
            src,
            attn_mask=src_attn_mask,
            key_padding_mask=src_key_padding_mask
        )
        src, st_attn_map = self.st_attn(
            src,
            tgt,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        return src, self_attn_map, st_attn_map


class Smoother(nn.Module):
    def __init__(self, params: AttributeDict):
        super().__init__()
        self.self_attn = SelfAttention(params, is_ffn=True)

    def forward(self,
                src: Tensor,
                src_attn_mask: Optional[Tensor],
                src_key_padding_mask: Optional[Tensor]
                ) -> Tuple[Tensor, Optional[Tensor]]:
        src, attn_map = self.self_attn(
            src,
            attn_mask=src_attn_mask,
            key_padding_mask=src_key_padding_mask
        )
        return src, attn_map


class PostNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        channel = params.model.channel
        self.net = nn.Sequential(
            Conv1d(params.mel_size, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, params.mel_size, 5),
            nn.BatchNorm1d(params.mel_size),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.net(x)
