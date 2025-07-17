import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=8, memory_blocks=256):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor))

        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Sigmoid())

        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):

        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)
        low_rank_f = self.subnet(y).unsqueeze(2)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)

        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2

        return out


class LRR_block(nn.Module):
    def __init__(self, num_feat, squeeze_factor=8, memory_blocks=256):
        super(LRR_block, self).__init__()
        self.num_feat = num_feat
        self.cab = nn.Sequential(
            ChannelAttention(num_feat, squeeze_factor, memory_blocks))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)
        out = self.cab(x)
        out = out.view(B, H, W, C)
        out = out.permute(0, 3, 1, 2)

        return out


