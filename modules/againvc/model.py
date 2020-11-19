from utils import normalize, get_wav_mel
from .networks import Encoder, Decoder
from ..base import BaseModel


class AgainVCModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, src, tgt):
        src, _ = self.encoder(src)
        _, (means, stds) = self.encoder(tgt)
        src = self.decoder(src, means, stds)
        return src

    def inference(self, src_path: str, tgt_path: str):
        mel_src, mel_tgt = self._preprocess(src_path, tgt_path)
        dec = self.forward(mel_src, mel_tgt)
        wav = self._mel_to_wav(dec)
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
