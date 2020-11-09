import torch.nn as nn

from .activations import Activation
from .layers import Conv1d, InstanceNorm1d, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_conv = Conv1d(params.mel_size, params.model.channel, 1)

        self.layers = nn.ModuleList([
            EncoderLayer(
                params.model.channel,
                params.model.channel,
                params.model.act_name
            ) for _ in range(params.model.n_enc_layers)
        ])
        self.norm = InstanceNorm1d()

        self.out_conv = Conv1d(params.model.channel, params.model.latent_channel, 1)
        self.act = Activation('v-sigmoid')

    def forward(self, x):
        means, stds = list(), list()

        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
            x, (m, s) = self.norm(x)
            means.append(m)
            stds.append(s)
        x = self.act(self.out_conv(x))
        return x, (means, stds)


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_conv = nn.Sequential(
            Conv1d(params.model.latent_channel, params.model.channel, 3),
            Activation(params.model.act_name)
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                params.model.channel,
                params.model.channel,
                params.model.act_name
            ) for _ in range(params.model.n_dec_layers)
        ])

        self.norm = InstanceNorm1d()

        self.rnn = nn.GRU(params.model.channel, params.model.channel, num_layers=2)
        self.linear = nn.Linear(params.model.channel, params.mel_size)

        self.n_layers = params.model.n_dec_layers

    def forward(self, x, means, stds):
        x = self.in_conv(x)

        for i in range(self.n_layers):
            x = self.layers[i](x)
            x, _ = self.norm(x)
            x = x * stds[i] + means[i]
        # (B, C, L) => (B, L, C)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.linear(x)
        # (B, L, C) => (B, C, L)
        x = x.transpose(1, 2)
        return x
