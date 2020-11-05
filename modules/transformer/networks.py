import torch.nn as nn

from .layers import Conv1d, SourceTargetAttention, SelfAttention


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_conv = Conv1d(params.mel_size, params.model.channel, 1)

        self.layers = nn.ModuleList([
            SelfAttention(params, is_ffn=True) for _ in range(params.model.n_layers)
        ])

    def forward(self, x):
        outputs = list()
        x = self.in_conv(x).permute(2, 0, 1)
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_layers = params.model.n_layers

        self.in_conv = Conv1d(params.mel_size, params.model.channel, 1)

        self.self_attns = nn.ModuleList([
            SelfAttention(params, is_ffn=False) for _ in range(params.model.n_layers)
        ])

        self.st_attns = nn.ModuleList([
            SourceTargetAttention(params, is_ffn=True) for _ in range(params.model.n_layers)
        ])
        self.linear = nn.Linear(params.model.channel, params.mel_size)

        channel = params.model.channel
        self.post_net = nn.Sequential(
            Conv1d(params.mel_size, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, channel, 5),
            nn.BatchNorm1d(channel),
            nn.Tanh(),
            nn.Dropout(0.5),
            Conv1d(channel, params.mel_size, 5),
            nn.BatchNorm1d(params.mel_size),
            nn.Dropout(0.5),
        )

    def forward(self, src, c_tgt):
        c_tgt = c_tgt[::-1]
        src = self.in_conv(src).permute(2, 0, 1)
        for i in range(self.n_layers):
            src = self.self_attns[i](src)
            src = self.st_attns[i](src, c_tgt[i])
        src = self.linear(src.permute(1, 0, 2)).transpose(1, 2)
        src = src + self.post_net(src)
        return src
