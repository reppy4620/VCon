import torch
from resemblyzer import VoiceEncoder

from ..common.model_base import AutoVCModelBase
from ..common.networks import AttnEncoder, Decoder, Postnet


class AttnAutoVCModel(AutoVCModelBase):
    def __init__(self, params):
        super().__init__()

        self.encoder = AttnEncoder(params.model.dim_neck, params.speaker_emb_dim)
        self.decoder = Decoder(params.model.dim_neck, params.speaker_emb_dim, params.model.dim_pre)
        self.postnet = Postnet()

        self.style_encoder = VoiceEncoder()

        self.vocoder = None

    def forward(self, raw, spec):

        c_src = self._make_speaker_vectors(raw, spec.size(-1), spec.device)

        codes, mel_outputs, mel_outputs_postnet = self._forward(spec, c_src)

        return (
            mel_outputs,  # decoder output
            mel_outputs_postnet,  # postnet output
            codes,  # encoder output
            self.encoder(mel_outputs_postnet, c_src)  # encoder output using postnet output
        )

    def _forward(self, spec, c_src, c_tgt=None):

        codes = self.encoder(spec, c_src)

        # (Batch, Mel-bin, Time) => (Batch, Time, Mel-bin) for LSTM
        decoder_input = torch.cat((codes, c_src if c_tgt is None else c_tgt), dim=1).transpose(1, 2)

        mel_outputs = self.decoder(decoder_input)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return codes, mel_outputs, mel_outputs_postnet
