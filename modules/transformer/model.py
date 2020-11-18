from typing import Tuple

from torch import Tensor

from utils import AttributeDict
from .networks import (
    ContentEncoder, SpeakerEncoder, Decoder
)
from ..base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, params: AttributeDict):
        super().__init__(params)

        self.content_encoder = ContentEncoder(params)
        self.speaker_encoder = SpeakerEncoder(params)
        self.decoder = Decoder(params)

    def forward(self, src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor]:
        src, q_loss = self.content_encoder(src)
        c_tgt = self.speaker_encoder(tgt)
        src = self.decoder(src, c_tgt)
        return src, q_loss

    def inference(self, src_path: str, tgt_path: str):
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        mel_out, _ = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(mel_out)
        return wav
