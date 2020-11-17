import torch.nn as nn

from .layers import Conv1d, Quantize, ConvExtractor, SelfAttention


class ContentEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv = ConvExtractor(params.mel_size, params.model.channel, params.model.emb_dim, params.model.n_ce)
        self.quantize = Quantize(params.model.emb_dim, params.model.n_emb)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, q_loss = self.quantize(x)
        x = x.transpose(1, 2)
        return x, q_loss


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv = nn.Sequential(
            Conv1d(params.speaker_emb_dim + params.model.emb_dim, params.speaker_emb_dim + params.model.emb_dim, 5),
            nn.BatchNorm1d(params.speaker_emb_dim + params.model.emb_dim),
            nn.GELU(),
            Conv1d(params.speaker_emb_dim + params.model.emb_dim, params.model.channel, 5),
            nn.BatchNorm1d(params.model.channel),
            nn.GELU(),
            Conv1d(params.model.channel, params.model.channel, 5),
            nn.BatchNorm1d(params.model.channel),
            nn.GELU(),
        )

        self.self_attns = nn.Sequential(*[
            SelfAttention(params, is_ffn=True) for _ in range(params.model.n_layers)
        ])
        self.linear = nn.Linear(params.model.channel, params.mel_size)

    def forward(self, src):
        src = self.conv(src)
        src = src.permute(2, 0, 1)
        src = self.self_attns(src)
        src = self.linear(src.permute(1, 0, 2)).transpose(1, 2)
        return src


class PostNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        channel = params.model.channel
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)
