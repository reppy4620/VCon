import torch
from resemblyzer import VoiceEncoder

from ..common.model_base import AutoVCModelBase
from ..common.networks import VQEncoder, Decoder, Postnet


class VQAutoVCModel(AutoVCModelBase):
    def __init__(self, params):
        super().__init__()

        self.encoder = VQEncoder(params.model.dim_neck, params.speaker_emb_dim, params.model.freq,
                                 params.model.n_embeddings, params.model.embedding_dim)
        self.decoder = Decoder(params.model.dim_neck, params.speaker_emb_dim, params.model.dim_pre)
        self.postnet = Postnet()

        self.style_encoder = VoiceEncoder()

        self.vocoder = None

    def forward(self, raw, spec):

        c_src = self._make_speaker_vectors(raw, spec.size(-1), spec.device)

        codes, vq_loss, mel_outputs, mel_outputs_postnet = self._forward(spec, c_src)

        return (
            mel_outputs,  # decoder output
            mel_outputs_postnet,  # postnet output
            torch.cat(codes, dim=-1),  # encoder output
            torch.cat(self.encoder(mel_outputs_postnet, c_src)[0], dim=-1),  # encoder output using postnet output
            vq_loss
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

        c_src = self._make_speaker_vectors([raw_src], spec_src.size(-1), spec_src.device)
        c_tgt = self._make_speaker_vectors([raw_tgt], spec_src.size(-1), spec_src.device)

        _, _, _, mel_outputs_postnet = self._forward(spec_src, c_src, c_tgt)

        wav = self._mel_to_wav(mel_outputs_postnet)
        return wav

    def _forward(self, spec, c_src, c_tgt=None):
        codes, vq_loss = self.encoder(spec, c_src)
        # almost equivalent to torch.nn.functional.interpolate
        code_exp = torch.cat(
            [c.unsqueeze(-1).expand(-1, -1, spec.size(-1) // len(codes)) for c in codes],
            dim=-1
        )

        # (Batch, Mel-bin, Time) => (Batch, Time, Mel-bin) for LSTM
        decoder_input = torch.cat([code_exp, c_src if c_tgt is None else c_tgt], dim=1).transpose(1, 2)

        mel_outputs = self.decoder(decoder_input)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return codes, vq_loss, mel_outputs, mel_outputs_postnet
