import torch
from resemblyzer import VoiceEncoder

from utils import normalize, get_wav_mel
from .networks import Encoder, Decoder
from ..base import BaseModel


class VQVCModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.speaker_encoder = VoiceEncoder()
        self.freeze(self.speaker_encoder)

    def forward(self, wavs, mels):
        emb = self._make_speaker_vectors(wavs, mels.device)
        q_afters, diff = self.encoder(mels)
        dec = self.decoder(q_afters, emb)
        return dec, diff

    def inference(self, src_path: str, tgt_path: str):
        wav_src, wav_tgt, mel_src = self._preprocess(src_path, tgt_path)

        emb = self._make_speaker_vectors([wav_tgt], mel_src.device)

        q_afters, _ = self.encoder(mel_src)
        dec = self.decoder(q_afters, emb)

        wav = self._mel_to_wav(dec)
        return wav

    def _make_speaker_vectors(self, wavs, device):
        c = [self.speaker_encoder.embed_utterance(x) for x in wavs]
        c = torch.tensor(c, dtype=torch.float, device=device)
        return c

    def _preprocess(self, src_path: str, tgt_path: str):
        wav_src, mel_src = get_wav_mel(src_path)
        wav_tgt, _ = get_wav_mel(tgt_path)
        mel_src = self._preprocess_mel(mel_src)
        return wav_src, wav_tgt, mel_src

    def _preprocess_mel(self, mel):
        if self.is_normalize:
            mel = normalize(mel)
        mel = self._adjust_length(mel, freq=4)
        mel = self.unsqueeze_for_input(mel)
        return mel
