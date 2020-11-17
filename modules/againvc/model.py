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
