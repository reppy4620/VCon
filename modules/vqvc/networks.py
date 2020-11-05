import torch.nn as nn

from .layers import (
    EncoderLayer, DecoderLayer, Quantize
)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        k1 = [9, 7, 5]
        k2 = [7, 5, 3]

        self.layers = nn.ModuleList([
            EncoderLayer(
                params.model.in_channel if i == 0 else params.model.channel // 2 ** i,
                params.model.channel // 2 ** (i+1),
                params.model.channel // 2 ** i,
                k1[i],
                k2[i]
            ) for i in range(3)
        ])
        self.quantize = nn.ModuleList([
            Quantize(
                params.model.channel // 2 ** (i+1),
                params.model.n_embed // 2 ** i
            ) for i in range(3)
        ])
        self.instance_norm = nn.ModuleList([
            nn.InstanceNorm1d(
                params.model.channel // 2 ** (i+1)
            ) for i in range(3)
        ])

    def forward(self, x):
        q_afters = []
        diff_total = 0

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x_ = self.instance_norm[i](x)
            q_after, diff = self.quantize[i](x_.transpose(1, 2))
            q_after = q_after.transpose(1, 2)

            q_afters.append(q_after)
            diff_total += diff
        return q_afters, diff_total


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        k1 = [8, 6, 4]
        k2 = [7, 5, 3]

        blocks = [
            DecoderLayer(
                params.model.channel // 2 ** (i+1),
                params.model.in_channel if i == 0 else params.model.channel // 2 ** i,
                params.model.channel // 2 ** i,
                params.speaker_emb_dim,
                k1[i],
                k2[i]
            ) for i in range(3)
        ]

        self.blocks = nn.ModuleList(blocks[::-1])

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(self, q_afters, emb):
        q_afters = q_afters[::-1]
        x = 0

        for i in range(len(self.blocks)):
            x = self.blocks[i](x+q_afters[i], emb)
        x = x + self.post_net(x)
        return x
