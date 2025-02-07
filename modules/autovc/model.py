import torch
from resemblyzer import VoiceEncoder

from utils import normalize, get_wav_mel
from ..base import BaseModel
from .networks import Encoder, Decoder, Postnet


class AutoVCModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.encoder = Encoder(params.model.dim_neck, params.speaker_emb_dim, params.model.freq)
        self.decoder = Decoder(params.model.dim_neck, params.speaker_emb_dim, params.model.dim_pre)
        self.postnet = Postnet()

        self.style_encoder = VoiceEncoder()
        self.freeze(self.style_encoder)

    def forward(self, wavs, mels):
        c_src = self._make_speaker_vectors(wavs, mels.size(-1), mels.device)

        codes, mel_outputs, mel_outputs_postnet = self._forward(mels, c_src)

        return (
            mel_outputs,  # decoder output
            mel_outputs_postnet,  # postnet output
            torch.cat(codes, dim=-1),  # encoder output
            torch.cat(self.encoder(mel_outputs_postnet, c_src), dim=-1)  # encoder output using postnet output
        )

    def inference(self, src_path: str, tgt_path: str):
        wav_src, wav_tgt, mel_src = self._preprocess(src_path, tgt_path)

        c_src = self._make_speaker_vectors([wav_src], mel_src.size(-1), mel_src.device)
        c_tgt = self._make_speaker_vectors([wav_tgt], mel_src.size(-1), mel_src.device)

        _, _, mel_outputs_postnet = self._forward(mel_src, c_src, c_tgt)

        wav = self._mel_to_wav(mel_outputs_postnet)
        return wav

    def _forward(self, mels, c_src, c_tgt=None):
        codes = self.encoder(mels, c_src)
        # almost equivalent to torch.modules.functional.interpolate
        code_exp = torch.cat(
            [c.unsqueeze(-1).expand(-1, -1, mels.size(-1) // len(codes)) for c in codes],
            dim=-1
        )

        # (Batch, Mel-bin, Time) => (Batch, Time, Mel-bin) for LSTM
        decoder_input = torch.cat((code_exp, c_src if c_tgt is None else c_tgt), dim=1).transpose(1, 2)

        mel_outputs = self.decoder(decoder_input)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return codes, mel_outputs, mel_outputs_postnet

    def _make_speaker_vectors(self, wavs, time_size, device):
        c = [self.style_encoder.embed_utterance(x) for x in wavs]
        c = torch.tensor(c, dtype=torch.float, device=device)
        c = c[:, :, None].expand(-1, -1, time_size)
        return c

    def _preprocess(self, src_path: str, tgt_path: str):
        wav_src, mel_src = get_wav_mel(src_path)
        wav_tgt, _ = get_wav_mel(tgt_path)
        mel_src = self._preprocess_mel(mel_src)
        return wav_src, wav_tgt, mel_src

    def _preprocess_mel(self, mel):
        if self.is_normalize:
            mel = normalize(mel)
        mel = self._adjust_length(mel, self.freq)
        mel = self.unsqueeze_for_input(mel)
        return mel
