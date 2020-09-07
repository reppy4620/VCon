import torch.nn as nn

from .networks import Encoder, Decoder, Postnet, VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.postnet = Postnet(params)

        self.vq = VectorQuantizer(params)

    def forward(self, x, c_src, c_tgt):
        # x.size() = (BatchSize, Length, EmbedSize)
        # c_src.size() = c_tgt.size() = (BatchSize, SpeakerEmbedSize)

        # z.size() = (BatchSize, Length, EmbedSize)
        z = self.encoder(x, c_src)
        # quantize.size() = (BatchSize, Length, EmbedSize)
        quantize, diff, _ = self.vq(z)
        # out_dec.size() = (BatchSize, Length, EmbedSize)
        out_dec = self.decoder(quantize, c_tgt)
        # out.size() = (BatchSize, Mel-Channel, Length)
        out = self.postnet(out_dec) + out_dec
        # out.size() = (BatchSize, Mel-Channel, Length)
        out = out.transpose(1, 2)
        return out, diff
