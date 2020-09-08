import torch
from resemblyzer import VoiceEncoder

from nn.base import ModelMixin
from .networks import QuartzEncoder, QuartzDecoder


# Non-autoregressive VC model using QuartzNet architecture
class QuartzModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.encoder = QuartzEncoder(params)
        self.decoder = QuartzDecoder(params)

        self.speaker_encoder = VoiceEncoder()

        self.vocoder = None

    def forward(self, raw, spec):
        # raw: List[np.array], np.array: (L,), L = Length of raw wav data
        # spec: (B, M, T), B = BatchSize, M = MelSize, T = Time in Spectrogram

        # embed_utterance is implemented for single wav data.
        c_src = list(map(self.speaker_encoder.embed_utterance, raw))
        c_src = torch.tensor(c_src, dtype=torch.float).to(spec.device)
        # expand for concatenating encoder output with c_src
        c_src = c_src.unsqueeze(-1).expand(-1, -1, spec.size(-1))

        out = self.encoder(spec)
        out = self.decoder(torch.cat([out, c_src], dim=1))
        return out

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        # generate speaker embedding
        c_tgt = self.speaker_encoder.embed_utterance(raw_tgt)
        c_tgt = torch.tensor(c_tgt, dtype=torch.float, device=spec_src.device)
        c_tgt = c_tgt[None, :, None].expand(-1, -1, spec_src.size(-1))

        out = self.encoder(spec_src)
        out = self.decoder(torch.cat([out, c_tgt], dim=1))
        wav = self.vocoder.inverse(out).squeeze(0).cpu().detach().numpy()
        return wav
