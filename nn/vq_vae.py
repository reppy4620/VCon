import torch.nn as nn

from .layers import VectorQuantizer
from .networks import Encoder, Decoder, Postnet


class VQVAE(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(params.encoder)
        self.decoder = Decoder(params.decoder)
        self.postnet = Postnet(params.postnet)

        self.vq = VectorQuantizer(params.vq)

    def forward(self, x, c_src, c_trg):
        z = self.encoder(x, c_src)
        quantize, diff, _ = self.vq(z)
        out_dec = self.decoder(quantize)
        out = self.postnet(out_dec.transpose(2, 1)).transpose(2, 1) + out_dec
        return out, diff
