import torch
import torch.nn as nn

from ..base import ModelMixin
from .layers import Quantize
from .networks import Decoder

from utils import denormalize


class VQVCModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(params.model.in_channel // 2 ** i, params.model.channel, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(params.model.channel, params.model.in_channel // 2 ** (i + 1), 3, 1, 1),
            ) for i in range(3)
        ])
        self.quantize = nn.ModuleList([
            Quantize(
                params.model.in_channel // 2 ** (i + 1),
                params.model.n_embed // 2 ** i
            ) for i in range(3)
        ])

        self.dec = Decoder(params.model.in_channel, params.model.channel, params.model.num_groups)

        self.vocoder = None

    def forward(self, raw, spec):
        enc, sp_embed, diff = self.encode(spec)
        dec = self.decode(enc, sp_embed)
        return dec, enc, diff

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        spec_tgt = self.vocoder(torch.tensor(raw_tgt, dtype=torch.float)[None, :]).to(spec_src.device)

        enc_src, _, _ = self.encode(spec_src)
        _, sp_embed_tgt, _ = self.encode(spec_tgt)

        dec = self.decode(enc_src, sp_embed_tgt)

        wav = self._mel_to_wav(dec)
        return wav

    def encode(self, x):
        q_after_block = []
        sp_embedding_block = []
        diff_total = 0

        for i, (enc_block, quantize) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)
            x_ = x - torch.mean(x, dim=2, keepdim=True)
            std_ = torch.norm(x_, dim=2, keepdim=True) + 1e-4
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim=1, keepdim=True)
            q_after, diff = quantize(x_.permute(0, 2, 1))
            q_after = q_after.permute(0, 2, 1)

            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim=1, keepdim=True) + 1e-4) / 3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
        return q_after_block, sp_embedding_block, diff_total

    def decode(self, quantized, sp):
        decoded = self.dec(quantized, sp)
        return decoded

    def _mel_to_wav(self, mel):
        mel = denormalize(mel)
        wav = self.vocoder.inverse(mel).squeeze(0).detach().cpu().numpy()
        return wav
