import torch

from nn.base import ModelMixin
from .network import Encoder, Decoder

from resemblyzer import VoiceEncoder


class AdaINVCModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.speaker_encoder = VoiceEncoder()

        self.vocoder = None

    def forward(self, raw, spec):
        cond = [self.speaker_encoder.embed_utterance(x) for x in raw]
        cond = torch.tensor(cond, dtype=torch.float, device=spec.device)

        enc = self.encoder(spec)
        dec = self.decoder(enc, cond)
        return dec

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        cond = [self.speaker_encoder.embed_utterance(x) for x in raw_tgt]
        cond = torch.tensor(cond, dtype=torch.float, device=spec_src.device)

        enc = self.encoder(spec_src)
        dec = self.decoder(enc, cond)

        return dec
