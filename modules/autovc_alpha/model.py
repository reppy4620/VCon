import torch
from resemblyzer import VoiceEncoder

from .networks import ContentEncoder, Decoder, PostNet
from ..base import BaseModel


class AutoVCAlphaModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.encoder = ContentEncoder(params)
        self.decoder = Decoder(params)
        self.postnet = PostNet(params)

        self.style_encoder = VoiceEncoder()
        self.freeze(self.style_encoder)

    def forward(self, wavs, mels):
        c_src = self._make_speaker_vectors(wavs, mels.size(-1), mels.device)

        codes, mel_outputs, mel_outputs_postnet, q_loss = self._forward(mels, c_src)

        return (
            mel_outputs,
            mel_outputs_postnet,
            codes,
            self.encoder(mel_outputs_postnet)[0],
            q_loss
        )

    def inference(self, src_path: str, tgt_path: str):
        wav_src, wav_tgt, mel_src = self._preprocess(src_path, tgt_path)

        c_src = self._make_speaker_vectors([wav_src], mel_src.size(-1), mel_src.device)
        c_tgt = self._make_speaker_vectors([wav_tgt], mel_src.size(-1), mel_src.device)

        _, _, mel_outputs_postnet, _ = self._forward(mel_src, c_src, c_tgt)

        wav = self._mel_to_wav(mel_outputs_postnet)
        return wav

    def _forward(self, mels, c_src, c_tgt=None):
        codes, q_loss = self.encoder(mels)

        decoder_input = torch.cat((codes, c_src if c_tgt is None else c_tgt), dim=1)

        mel_outputs = self.decoder(decoder_input)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return codes, mel_outputs, mel_outputs_postnet, q_loss

    def _make_speaker_vectors(self, wavs, time_size, device):
        c = [self.style_encoder.embed_utterance(x) for x in wavs]
        c = torch.tensor(c, dtype=torch.float, device=device)
        c = c[:, :, None].expand(-1, -1, time_size)
        return c
