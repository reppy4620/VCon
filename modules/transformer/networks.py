import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv1d, Quantize, ConvExtractor, SourceTargetAttention, SelfAttention


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


class SpeakerEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_layers = params.model.n_layers

        self.layers = nn.ModuleList([Conv1d(params.mel_size, params.model.channel, 5)])

        for _ in range(params.model.n_layers-1):
            self.layers.append(Conv1d(params.model.channel, params.model.channel, 5))

    def forward(self, x):
        outputs = list()
        for i in range(self.n_layers):
            x = self.layers[i](x)
            outputs.append(x.permute(2, 0, 1))
            if i != self.n_layers-1:
                x = F.gelu(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_layers = params.model.n_layers

        self.adjust_conv = nn.Sequential(
            Conv1d(params.model.emb_dim, params.model.channel, 3),
            nn.BatchNorm1d(params.model.channel),
            nn.GELU(),
            Conv1d(params.model.channel, params.model.channel, 3),
            nn.BatchNorm1d(params.model.channel),
            nn.GELU(),
        )

        self.self_attns = nn.ModuleList([
            SelfAttention(params, is_ffn=False) for _ in range(params.model.n_layers)
        ])

        self.st_attns = nn.ModuleList([
            SourceTargetAttention(params, is_ffn=True) for _ in range(params.model.n_layers)
        ])

        self.smoothers = nn.Sequential(*[
            SelfAttention(params, is_ffn=True) for _ in range(params.model.n_layers)
        ])
        self.linear = nn.Linear(params.model.channel, params.mel_size)

        self.post_net = PostNet(params)

    def forward(self, src, c_tgt):
        c_tgt = c_tgt[::-1]
        src = self.adjust_conv(src)
        src = src.permute(2, 0, 1)
        for i in range(self.n_layers):
            src = self.self_attns[i](src)
            src = self.st_attns[i](src, c_tgt[i])
        src = self.smoothers(src)
        src = self.linear(src.permute(1, 0, 2)).transpose(1, 2)
        src = src + self.post_net(src)
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

