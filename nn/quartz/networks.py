import torch.nn as nn

from .layers import QuartzBlock


class QuartzEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_layer = QuartzBlock(
            params.mel_size,
            params.encoder.c_ins[0],
            params.encoder.kernels[0],
            params.encoder.strides[0],
            params.encoder.pads[0],
            params.encoder.n_block
        )

        self.layers = nn.ModuleList()
        for i in range(1, params.encoder.n_layers):
            self.layers.append(
                QuartzBlock(
                    params.encoder.c_ins[i],
                    params.encoder.c_outs[i],
                    params.encoder.kernels[i],
                    params.encoder.strides[i],
                    params.encoder.pads[i],
                    params.encoder.n_block
                )
            )

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class QuartzDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_layer = QuartzBlock(
            params.decoder.c_ins[0] + params.speaker_emb_dim,
            params.decoder.c_outs[0],
            params.decoder.kernels[0],
            params.decoder.strides[0],
            params.decoder.pads[0],
            params.decoder.n_block
        )
        self.layers = nn.ModuleList()
        for i in range(1, params.decoder.n_layers):
            self.layers.append(
                QuartzBlock(
                    params.decoder.c_ins[i],
                    params.decoder.c_outs[i],
                    params.decoder.kernels[i],
                    params.decoder.strides[i],
                    params.decoder.pads[i],
                    params.decoder.n_block
                )
            )

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class QuartzPostNet(QuartzDecoder):
    def __init__(self, params):
        super().__init__(params)
        self.in_layer = QuartzBlock(
            params.mel_size,
            params.decoder.c_outs[0],
            params.decoder.kernels[0],
            params.decoder.strides[0],
            params.decoder.pads[0],
            params.decoder.n_block
        )
