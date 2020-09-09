###################################################################
# Following models almost adopted from author's implementation
# [link] => https://github.com/auspicious3000/autovc
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvNorm, LinearNorm


class ContentEncoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    in_channels=80 + dim_emb if i == 0 else 512,
                    out_channels=512,
                    kernel_size=5, stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_src):
        x = torch.cat((x, c_src), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes


class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)

        self.convolutions = nn.ModuleList()
        for i in range(3):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=dim_pre,
                        out_channels=dim_pre,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                        w_init_gain='relu'),
                    nn.BatchNorm1d(dim_pre)
                )
            )

        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear_projection(outputs).transpose(1, 2)

        return decoder_output


class Postnet(nn.Module):

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=80,
                    out_channels=512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='tanh'
                ),
                nn.BatchNorm1d(512)
            )
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=512,
                        out_channels=512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                        w_init_gain='tanh'
                    ),
                    nn.BatchNorm1d(512)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=512,
                    out_channels=80,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='linear'
                ),
                nn.BatchNorm1d(80)
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x
