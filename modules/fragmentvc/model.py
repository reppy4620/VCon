import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Tuple, List

from utils import AttributeDict, get_wav_mel
from .layers import Conv1d
from .networks import (
    Extractor, Smoother, PostNet
)
from ..base import ModelMixin


class FragmentVCModel(ModelMixin):
    def __init__(self, params: AttributeDict):
        super().__init__()

        channel = params.model.channel

        self.pre_net = nn.Sequential(
            nn.Linear(params.model.in_channel, params.model.in_channel),
            nn.GELU(),
            nn.Linear(params.model.in_channel, channel)
        )

        self.conv1 = Conv1d(params.mel_size, channel, 3, padding_mode='replicate')
        self.conv2 = Conv1d(channel, channel, 3, padding_mode='replicate')
        self.conv3 = Conv1d(channel, channel, 3, padding_mode='replicate')

        self.extractor1 = Extractor(params)
        self.extractor2 = Extractor(params)
        self.extractor3 = Extractor(params)

        self.smoothers = nn.ModuleList([Smoother(params) for _ in range(params.model.n_smoother)])

        self.linear = nn.Linear(channel, 80)
        self.post_net = PostNet(params)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        # src: (B, L, C)
        src = self.pre_net(src)
        # src: (L, B, C)
        src = src.transpose(0, 1)

        # tgt*: (B, C, L)
        tgt1 = self.conv1(tgt)
        tgt2 = self.conv2(F.gelu(tgt1))
        tgt3 = self.conv3(F.gelu(tgt2))

        # src: (L, B, C)
        src, _, attn_map1 = self.extractor1(
            src,
            tgt3.permute(2, 0, 1),  # (L, B, C)
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask
        )

        src, _, attn_map2 = self.extractor1(
            src,
            tgt2.permute(2, 0, 1),  # (L, B, C)
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask
        )

        src, _, attn_map3 = self.extractor1(
            src,
            tgt1.permute(2, 0, 1),  # (L, B, C)
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask
        )

        for smoother in self.smoothers:
            src, _ = smoother(src, src_key_padding_mask=src_mask)

        # src: (L, B, C) => (B, C, L)
        src = self.linear(src.permute(1, 0, 2)).transpose(1, 2)
        src = src + self.post_net(src)
        # src: (B, C, L)
        return src, [attn_map1, attn_map2, attn_map3]

    def inference(self, src_path: str, tgt_path: str):
        self._load_vocoder()
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        mel_out = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(mel_out)
        return wav

    def _preprocess(self, src_path: str, tgt_path: str):
        _, mel_src = get_wav_mel(src_path)
        _, mel_tgt = get_wav_mel(tgt_path)
        mel_src, mel_tgt = self._preprocess_mel(mel_src), self._preprocess_mel(mel_tgt)
        return mel_src, mel_tgt

    def _preprocess_mel(self, mel):
        mel = self.unsqueeze_for_input(mel)
        return mel
