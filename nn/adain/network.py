import torch
import torch.nn as nn

from .layers import EncoderLayer, DecoderLayer, DisLayer


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv_bank = nn.ModuleList([
            nn.Conv1d(
                params.mel_size,
                params.encoder.bank_channel,
                kernel_size=k,
                padding=(k-1)//2,
                padding_mode='reflect'
            ) for k in range(3, params.encoder.n_bank*2 + 3, 2)
        ])

        self.in_layer = nn.Sequential(
            nn.Conv1d(
                params.encoder.bank_channel*params.encoder.n_bank,
                params.encoder.channel,
                params.encoder.kernel_size,
                padding=(params.encoder.kernel_size-1)//2,
                padding_mode='reflect'
            ),
            nn.InstanceNorm1d(params.encoder.channel, affine=True),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                c=params.encoder.channel,
                ks=params.encoder.kernel_size
            ) for _ in range(params.encoder.n_layers-1)
        ])

    def forward(self, x):
        xs = [conv(x) for conv in self.conv_bank]
        x = torch.cat(xs, dim=1)

        x = self.in_layer(x)
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                c=params.decoder.channel,
                ks=params.decoder.kernel_size,
                speaker_emb_dim=params.speaker_emb_dim
            ) for _ in range(1, params.decoder.n_layers)
        ])
        self.out_layer = nn.Sequential(
            nn.Conv1d(
                params.decoder.channel,
                params.mel_size,
                params.decoder.kernel_size,
                padding=(params.encoder.kernel_size-1)//2,
                padding_mode='reflect'
            ),
            nn.InstanceNorm1d(params.mel_size, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        x = self.out_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.layers = nn.ModuleList([
            DisLayer(
                params.discriminator.channel*i,
                params.discriminator.channel*i*2,
                params.discriminator.kernel_size,
            ) for i in range(1, params.discriminator.n_layers)
        ])
        self.avg_layer = nn.AdaptiveAvgPool1d(1)
        self.out_layer = nn.Conv1d(params.discriminator.channel*(params.discriminator.n_layers-1), 1, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avg_layer(x)
        x = self.out_layer(x)
        return x
