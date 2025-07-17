import torch
import torch.nn as nn


class Residual_block(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1),
               nn.BatchNorm2d(in_ch),
               nn.LeakyReLU(inplace=True),
               nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class local_branch(nn.Module):
    def __init__(self, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, padding_mode='reflect'),
               nn.BatchNorm2d(in_ch),
               nn.LeakyReLU(inplace=True)]
        ly += [Residual_block(in_ch) for _ in range(num_module)]
        self.body = nn.Sequential(*ly)
        self.Conv_out = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, x):
        out = self.body(x)
        local_feature = self.Conv_out(out)

        return local_feature
