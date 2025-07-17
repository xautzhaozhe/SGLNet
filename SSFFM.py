import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 3, padding=1, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention2(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention2, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
        self.ma = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

        self.Conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        x_gap = self.gap(x)
        x_map = self.map(x)
        cattn = self.ca(x_gap)
        mattn = self.ma(x_map)
        chan_out = self.Conv(cattn + mattn)
        return chan_out


class SFFM(nn.Module):
    def __init__(self, dim, reduction=8):
        super(SFFM, self).__init__()
        self.sa = SpatialAttention()
        self.convfusion = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect')
        self.ca = ChannelAttention2(dim, reduction)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, LRR_feat, local_feat, graph_feat):

        lrr_feat = self.convfusion(LRR_feat)
        initial = lrr_feat + local_feat + graph_feat
        sattn = self.sa(initial)
        spa_out = initial * sattn
        spettn = self.ca(spa_out)
        spe_out = spettn * spa_out

        pattn2 = self.sigmoid(spe_out)

        result = pattn2 * local_feat + (1 - pattn2) * graph_feat
        result = self.conv(result)

        return result








