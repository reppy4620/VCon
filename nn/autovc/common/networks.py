###################################################################
# Following models almost adopted from author's implementation
# [link] => https://github.com/auspicious3000/autovc
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvNorm, LinearNorm, AttnBlock, VQEmbeddingEMA


class NormalEncoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=80 + dim_emb if i == 0 else 512,
                    out_channels=512,
                    kernel_size=5, stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(512, affine=True)
            ) for i in range(3)
        ])

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

        # Author's implementation is different from paper.
        # In paper, author uses time steps {0, 32, 64, ...} for forward output
        # and for backward output, he uses time steps {31, 63, 95, ...},
        # but in github repo, he uses {31, 63, 95, ...} for former and {0, 32, 64, ...} for latter.
        # Therefore, I changed this part to match the paper.
        codes = [
            torch.cat(
                (out_forward[:, i, :], out_backward[:, i + self.freq - 1, :]),
                dim=-1
            ) for i in range(0, outputs.size(1), self.freq)
        ]

        return codes


class AttnEncoder(nn.Module):
    def __init__(self, dim_neck, dim_emb):
        super().__init__()
        self.dim_neck = dim_neck

        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=80 + dim_emb if i == 0 else 512,
                    out_channels=512,
                    kernel_size=5, stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(512, affine=True)
            ) for i in range(3)
        ])

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)
        self.attn_forward = AttnBlock(dim_neck)
        self.attn_backward = AttnBlock(dim_neck)

    def forward(self, x, c_src):
        x = torch.cat((x, c_src), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck].transpose(1, 2)
        out_backward = outputs[:, :, self.dim_neck:].transpose(1, 2)

        codes = torch.cat((self.attn_forward(out_forward), self.attn_backward(out_backward)), dim=1)

        return codes


class VQEncoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq, n_embeddings, embedding_dim):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=80 + dim_emb if i == 0 else 512,
                    out_channels=512,
                    kernel_size=5, stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(512, affine=True)
            ) for i in range(3)
        ])

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)
        self.vq_forward = VQEmbeddingEMA(n_embeddings, embedding_dim)
        self.vq_backward = VQEmbeddingEMA(n_embeddings, embedding_dim)

    def forward(self, x, c_src):
        x = torch.cat((x, c_src), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes_f, loss_f = self.vq_forward(out_forward)
        codes_b, loss_b = self.vq_backward(out_backward)
        codes = [
            torch.cat(
                (codes_f[:, i, :], codes_b[:, i + self.freq - 1, :]),
                dim=-1
            ) for i in range(0, outputs.size(1), self.freq)
        ]

        return codes, loss_f + loss_b


class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)

        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=dim_pre,
                    out_channels=dim_pre,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre, affine=True)
            ) for _ in range(3)
        ])

        self.lstm2 = nn.LSTM(dim_pre, dim_pre, 2, batch_first=True)

        self.linear_projection = LinearNorm(dim_pre, 80)

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
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=80 if i == 0 else 512,
                    out_channels=512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain='tanh'
                ),
                nn.BatchNorm1d(512, affine=True)
            ) for i in range(5-1)
        ])

        # paper indicates that bn and act is not applied after last convolution
        self.convolutions.append(
            ConvNorm(
                in_channels=512,
                out_channels=80,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
                w_init_gain='linear'
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x
