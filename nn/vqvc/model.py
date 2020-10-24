import torch
from resemblyzer import VoiceEncoder

from ..base import ModelMixin
from .networks import Encoder, Decoder

from utils import denormalize


class VQVCModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.speaker_encoder = VoiceEncoder()

        self.vocoder = None

        self.freeze(self.speaker_encoder)

    def forward(self, raw, spec):
        emb = self._make_speaker_vectors(raw, spec.device)
        enc, q_afters, diff = self.encoder(spec)
        dec = self.decoder(enc, q_afters, emb)
        return dec, diff

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        emb = self.speaker_encoder.embed_utterance(raw_tgt)

        enc, q_afters, _ = self.encoder(spec_src)
        dec = self.decoder(enc, q_afters, emb)

        wav = self._mel_to_wav(dec)
        return wav

    def _mel_to_wav(self, mel):
        mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).squeeze(0).detach().cpu().numpy()
        return wav

    def _make_speaker_vectors(self, raw, device):
        c = [self.speaker_encoder.embed_utterance(x) for x in raw]
        c = torch.tensor(c, dtype=torch.float, device=device)
        return c
