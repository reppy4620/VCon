import torch
import torch.nn as nn
from resemblyzer import VoiceEncoder

from .vq_vae import VQVAE


class VCModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.vq_vae = VQVAE(params)
        self.speaker_encoder = VoiceEncoder()

    def forward(self, raw, spec):
        # embed_utterance is implemented for single wav data.
        c_src = list(map(self.speaker_encoder.embed_utterance, raw))
        c_src = torch.tensor(c_src, dtype=torch.float)
        out, diff = self.vq_vae(spec, c_src, c_src)
        return out, diff

    def generate(self, raw_src, raw_tgt, spec_src):
        c_src = self.speaker_encoder.embed_utterance(raw_src)
        c_tgt = self.speaker_encoder.embed_utterance(raw_tgt)
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")
        out, _ = self.vq_vae(spec_src, c_src, c_tgt)
        return out
