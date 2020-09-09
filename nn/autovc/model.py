import torch
from resemblyzer import VoiceEncoder

from .networks import ContentEncoder, Decoder, Postnet
from nn.base import ModelMixin


class AutoVCModel(ModelMixin):
    def __init__(self, params):
        super().__init__()

        self.encoder = ContentEncoder(params.model.dim_neck, params.speaker_emb_dim, params.model.freq)
        self.decoder = Decoder(params.model.dim_neck, params.speaker_emb_dim, params.model.dim_pre)
        self.postnet = Postnet()

        self.style_encoder = VoiceEncoder()

        self.vocoder = None

    def forward(self, raw, spec):

        c_src = [self.style_encoder.embed_utterance(x) for x in raw]
        c_src = torch.tensor(c_src, dtype=torch.float, device=spec.device)
        c_src = c_src[:, :, None].expand(-1, -1, spec.size(-1))

        codes = self.encoder(spec, c_src)

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(spec.size(-1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        # (Batch, Mel-bin, Time) => (Batch, Time, Mel-bin) for LSTM
        encoder_outputs = torch.cat((code_exp, c_src), dim=1).transpose(1, 2)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return (
            mel_outputs,  # decoder output
            mel_outputs_postnet,  # postnet output
            torch.cat(codes, dim=-1),  # encoder output
            torch.cat(self.encoder(mel_outputs_postnet, c_src), dim=-1)  # encoder output using postnet output
        )

    def inference(self, raw_src, raw_tgt, spec_src):
        if self.vocoder is None:
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # if spec doesn't batch dim, unsqueeze spec
        if len(spec_src.size()) == 2:
            spec_src = spec_src.unsqueeze(0)
        # spec_src's size must be (Batch, Mel-bin, Time) or (Mel-bin, Time)
        elif len(spec_src.size()) != 3:
            raise ValueError("len(spec_src.size()) must be 2 or 3")

        c_src = self.style_encoder.embed_utterance(raw_src)
        c_src = torch.tensor(c_src, dtype=torch.float, device=spec_src.device)
        c_src = c_src[None, :, None].expand(-1, -1, spec_src.size(-1))

        c_tgt = self.style_encoder.embed_utterance(raw_tgt)
        c_tgt = torch.tensor(c_tgt, dtype=torch.float, device=spec_src.device)
        c_tgt = c_tgt[None, :, None].expand(-1, -1, spec_src.size(-1))

        codes = self.encoder(spec_src, c_src)

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(spec_src.size(-1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        # (Batch, Mel-bin, Time) => (Batch, Time, Mel-bin) for LSTM
        encoder_outputs = torch.cat((code_exp, c_tgt), dim=1).transpose(1, 2)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs_postnet = torch.log1p(mel_outputs_postnet)
        wav = self.vocoder.inverse(mel_outputs_postnet).squeeze(0).cpu().detach().numpy()
        return wav
