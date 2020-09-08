import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.vq_vae_autovc.layers import ConvNorm, LinearNorm


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.speaker_conv = ConvNorm(
            params.speaker_emb_dim,
            params.speaker_emb_dim // 4,
            bias=False
        )

        self.convolutions = nn.ModuleList()
        for i in range(params.encoder.n_conv):
            conv_layer = nn.Sequential(
                ConvNorm(params.mel_size+params.speaker_emb_dim // 4 if i == 0 else params.encoder.channel,
                         params.encoder.channel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                         w_init_gain='relu'),
                nn.BatchNorm1d(256)
            )
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(params.encoder.channel, params.emb_dim, 2, batch_first=True, bidirectional=True)

        self.out = ConvNorm(params.emb_dim * 2, params.emb_dim)

    def forward(self, x, c_src):

        c_src = self.speaker_conv(c_src.unsqueeze(-1)).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_src), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        out = self.out(x.transpose(1, 2)).transpose(1, 2)
        return out


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.speaker_conv = ConvNorm(
            params.speaker_emb_dim,
            params.speaker_emb_dim // 4,
            bias=False
        )

        self.lstm1 = nn.LSTM(
            params.emb_dim+params.speaker_emb_dim // 4,
            params.decoder.channel,
            num_layers=1,
            batch_first=True
        )

        convolutions = []
        for i in range(params.decoder.n_conv):
            conv_layer = nn.Sequential(
                ConvNorm(params.decoder.channel,
                         params.decoder.channel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(params.decoder.channel)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(params.decoder.channel, params.decoder.channel*2, 2, batch_first=True)

        self.linear_projection = LinearNorm(params.decoder.channel*2, params.mel_size)

    def forward(self, x, c_tgt):

        c_tgt = self.speaker_conv(c_tgt.unsqueeze(-1)).expand(-1, -1, x.size(1)).transpose(1, 2)
        x = torch.cat([x, c_tgt], dim=-1)

        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))
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
                ConvNorm(params.mel_size, params.postnet.channel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(params.postnet.channel))
        )

        for i in range(1, params.postnet.n_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(params.postnet.channel,
                             params.postnet.channel,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(params.postnet.channel))
            )

        self.convolutions.append(
            ConvNorm(params.postnet.channel, params.mel_size,
                     kernel_size=5, stride=1,
                     padding=2,
                     dilation=1,
                     w_init_gain='linear')
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convolutions[:-1]:
            x = torch.tanh(conv(x))

        x = self.convolutions[-1](x).transpose(1, 2)

        return x


# from https://github.com/vsimkus/voice-conversion
class VectorQuantizer(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.emb_dim = params.emb_dim
        self.n_embed = params.n_embed
        self.decay = params.ema.decay
        self.eps = params.ema.eps

        embed = torch.empty(self.emb_dim, self.n_embed).uniform_(-1./self.n_embed, 1./self.n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.emb_dim)
        # calculate distance
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.argmin(1)
        # change to one-hot and reshape for look-up
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.t())

        # EMA(Exponential Moving Average)
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind
