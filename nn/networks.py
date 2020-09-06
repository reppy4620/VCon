import torch
import torch.nn as nn

from .functions import tanhexp
from .layers import ConvNorm, LinearNorm


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.channel = params.channel
        self.freq = params.freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+params.speaker_emb_dim if i == 0 else params.channel,
                         params.channel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                         w_init_gain='relu'),
                nn.BatchNorm1d(256)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(params.channel, params.emb_dim, 2, batch_first=True, bidirectional=True)

        self.out = ConvNorm(params.emb_dim * 2, params.emb_dim)

    def forward(self, x, c_src):

        x = x.squeeze(1).transpose(2, 1)
        c_src = c_src.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_src), dim=1)

        for conv in self.convolutions:
            x = tanhexp(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        out = self.out(x)
        return out


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(params.emb_dim+params.speaker_emb_dim, params.channel, 1, batch_first=True)

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(params.channel,
                         params.channel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(params.channel)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(params.channel, params.channel*2, 2, batch_first=True)

        self.linear_projection = LinearNorm(params.channel*2, 80)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = tanhexp(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Postnet(nn.Module):
    def __init__(self, params):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, params.dim,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(params.dim,
                             params.dim,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(params.dim, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
        )

    def forward(self, x):
        for conv in self.convolutions[:-1]:
            x = torch.tanh(conv(x))

        x = self.convolutions[-1](x)

        return x
