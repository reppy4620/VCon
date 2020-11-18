from typing import Tuple

import torch.nn.functional as F
from torch import Tensor

from utils import AttributeDict
from .networks import (
    ContentEncoder, SpeakerEncoder, Decoder
)
from ..base import BaseModel


class TransformerAlphaModel(BaseModel):
    def __init__(self, params: AttributeDict):
        super().__init__(params)

        self.content_encoder = ContentEncoder(params)
        self.speaker_encoder = SpeakerEncoder(params)
        self.decoder = Decoder(params)

    def forward(self, src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        enc, q_loss = self.content_encoder(src)
        c_tgt = self.speaker_encoder(tgt)
        dec = self.decoder(enc, c_tgt)
        c_loss = F.l1_loss(enc, self.content_encoder(dec))
        return dec, q_loss, c_loss

    def inference(self, src_path: str, tgt_path: str):
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        mel_out, _, _ = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(mel_out)
        return wav
