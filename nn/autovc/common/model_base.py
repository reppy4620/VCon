import torch

from nn.base import ModelMixin
from utils import denormalize


class AutoVCModelBase(ModelMixin):
    def __init__(self):
        super().__init__()

        # following variable is initialized in inherits class
        self.style_encoder = None
        self.vocoder = None

    def forward(self, raw, spec):
        raise NotImplementedError

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        c_src = self._make_speaker_vectors([raw_src], spec_src.size(-1), spec_src.device)
        c_tgt = self._make_speaker_vectors([raw_tgt], spec_src.size(-1), spec_src.device)

        _, _, mel_outputs_postnet = self._forward(spec_src, c_src, c_tgt)

        wav = self._mel_to_wav(mel_outputs_postnet)
        return wav

    def _forward(self, spec, c_src, c_tgt=None):
        raise NotImplementedError

    def _make_speaker_vectors(self, raw, time_size, device):
        c = [self.style_encoder.embed_utterance(x) for x in raw]
        c = torch.tensor(c, dtype=torch.float, device=device)
        c = c[:, :, None].expand(-1, -1, time_size)
        return c

    def _mel_to_wav(self, mel):
        mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).squeeze(0).detach().cpu().numpy()
        return wav
