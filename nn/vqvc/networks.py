import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GBlock


class Decoder(nn.Module):
    def __init__(self, in_channel, channel, num_groups=4):
        super().__init__()

        blocks = [GBlock(in_channel // 2 ** i, in_channel // 2 ** i, channel, num_groups) for i in range(1, 4)]
        blocks_refine = [GBlock(in_channel // 2 ** i, in_channel // 2 ** i, channel, num_groups) for i in range(1, 4)]
        res_block = [GBlock(in_channel // 2 ** i, in_channel // 2 ** i, channel, num_groups) for i in range(1, 4)]

        self.blocks = nn.ModuleList(blocks[::-1])
        self.blocks_refine = nn.ModuleList(blocks_refine[::-1])
        self.res_block = nn.ModuleList(res_block[::-1])

        self.scale_factors = [2, 2, 2]

    def forward(self, q_after, sp_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        x = 0

        for i in range(len(self.blocks)):
            x = x + self.res_block[i](q_after[i] + sp_embed[i])
            x = F.interpolate(x, scale_factor=self.scale_factors[i], mode='nearest')
            x = x + self.blocks[i](x)
            x = torch.cat([x, x + self.blocks_refine[i](x)], dim=1)
        return x
