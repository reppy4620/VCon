import torch.nn as nn
from torch import Tensor

from utils import AttributeDict, normalize, get_wav_mel
from .layers import (
    SourceTargetAttention, SelfAttention,
    ConvExtractor, Conv1d
)
from ..base import ModelMixin


class TransformerModel(ModelMixin):
    def __init__(self, params: AttributeDict):
        super().__init__()

        channel = params.model.channel

        self.source_extractor = ConvExtractor(params)
        self.target_extractor = ConvExtractor(params)

        self.conv_layers = nn.ModuleList([
            Conv1d(
                channel,
                channel,
                3
            ) for _ in range(params.model.n_st_attn)
        ])

        self.conv1 = Conv1d(channel, channel, 3)
        self.conv2 = Conv1d(channel, channel, 3)
        self.conv3 = Conv1d(channel, channel, 3)

        self.st_attn_layers = nn.ModuleList([
            SourceTargetAttention(params) for _ in range(params.model.n_st_attn)
        ])

        self.st_attn1 = SourceTargetAttention(params)
        self.st_attn2 = SourceTargetAttention(params)
        self.st_attn3 = SourceTargetAttention(params)

        self.smoothers = nn.Sequential(*[
            SelfAttention(params) for _ in range(params.model.n_self_attn)
        ])

        self.post_net = nn.Sequential(
            nn.Conv1d(params.mel_size, channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(channel, channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(channel, channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(channel, channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(channel, params.mel_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

        self.n_st_attn = params.model.n_st_attn

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src, tgt = self.source_extractor(src), self.target_extractor(tgt)

        for i in range(self.n_st_attn):
            tgt = self.conv_layers[i](tgt)
            src = self.st_attn_layers[i](tgt, src)
        src = src + self.post_net(src)
        return src

    def inference(self, src_path: str, tgt_path: str):
        self._load_vocoder()
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        mel_out = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(mel_out)
        return wav

    def _preprocess(self, src_path: str, tgt_path: str):
        _, mel_src = get_wav_mel(src_path, to_mel=self.vocoder)
        _, mel_tgt = get_wav_mel(tgt_path, to_mel=self.vocoder)
        mel_src, mel_tgt = self._preprocess_mel(mel_src), self._preprocess_mel(mel_tgt)
        return mel_src, mel_tgt

    def _preprocess_mel(self, mel):
        mel = normalize(mel)
        mel = self.unsqueeze_for_input(mel)
        return mel
