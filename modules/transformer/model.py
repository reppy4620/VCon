from torch import Tensor
from typing import Tuple
from utils import AttributeDict, get_wav_mel, normalize, denormalize
from .networks import (
    ContentEncoder, SpeakerEncoder, Decoder
)
from ..base import ModelMixin


class TransformerModel(ModelMixin):
    def __init__(self, params: AttributeDict):
        super().__init__()

        self.content_encoder = ContentEncoder(params)
        self.speaker_encoder = SpeakerEncoder(params)
        self.decoder = Decoder(params)

        self.vocoder = None
        self._load_vocoder()

        self.is_normalize = params.is_normalize

    def forward(self, src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor]:
        src, q_loss = self.content_encoder(src)
        c_tgt = self.speaker_encoder(tgt)
        src = self.decoder(src, c_tgt)
        return src, q_loss

    def inference(self, src_path: str, tgt_path: str):
        self._load_vocoder()
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        mel_out, _ = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(mel_out)
        return wav

    def inverse(self, mel: Tensor):
        if len(mel.size()) == 2:
            mel = mel.unsqueeze(0)
        if self.is_normalize:
            mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).detach().cpu()
        return wav

    def _preprocess(self, src_path: str, tgt_path: str):
        _, mel_src = get_wav_mel(src_path)
        _, mel_tgt = get_wav_mel(tgt_path)
        mel_src, mel_tgt = self._preprocess_mel(mel_src), self._preprocess_mel(mel_tgt)
        return mel_src, mel_tgt

    def _preprocess_mel(self, mel):
        if self.is_normalize:
            mel = normalize(mel)
        mel = self.unsqueeze_for_input(mel)
        return mel
