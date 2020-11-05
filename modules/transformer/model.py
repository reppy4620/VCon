from torch import Tensor

from utils import AttributeDict, normalize, get_wav_mel
from .networks import (
    Encoder, Decoder
)
from ..base import ModelMixin


class TransformerModel(ModelMixin):
    def __init__(self, params: AttributeDict):
        super().__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        c_tgt = self.encoder(tgt)
        src = self.decoder(src, c_tgt)
        return src

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
        # mel = normalize(mel)
        mel = self.unsqueeze_for_input(mel)
        return mel
