import torch
from resemblyzer import VoiceEncoder

from nn.base import ModelMixin
from nn.vq_vae_autovc.vq_vae import VQVAE


class AutoVCBaseVQVAEModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.vq_vae = VQVAE(params)
        self.speaker_encoder = VoiceEncoder()
        self.vocoder = None

    def forward(self, raw, spec):
        # embed_utterance is implemented for single wav data.
        c_src = list(map(self.speaker_encoder.embed_utterance, raw))
        c_src = torch.tensor(c_src, dtype=torch.float).to(spec.device)
        out, diff = self.vq_vae(spec, c_src, c_src)
        return out, diff

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        # make src speaker embedding and tgt speaker embedding respectively
        c_src = self.speaker_encoder.embed_utterance(raw_src)
        c_tgt = self.speaker_encoder.embed_utterance(raw_tgt)
        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")
        # c_src.size = c_tgt.size = (256,)
        # so unsqueeze for adjusting to dimension
        c_src = torch.tensor(c_src, dtype=torch.float).unsqueeze(0)
        c_tgt = torch.tensor(c_tgt, dtype=torch.float).unsqueeze(0)
        out, _ = self.vq_vae(spec_src, c_src, c_tgt)
        wav = self.vocoder.inverse(out).squeeze(0).cpu().detach().numpy()
        return wav
