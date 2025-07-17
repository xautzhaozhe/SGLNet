import torch.nn as nn
from Spectral_LR import LRR_block
from Local_branch import local_branch
from Graph_module import Graph_branch
from SSFFM import SFFM


class SGLNet(nn.Module):

    def __init__(self, in_ch=189, out_ch=189, head_ch=48, local_blc=2, sq=10, mb=64):
        super().__init__()

        assert head_ch % 2 == 0, "base channel should be divided with 2"
        n1 = int(in_ch/2)

        # Encoder
        self.Encoder1 = nn.Sequential(nn.Conv2d(in_ch, n1, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU(), )

        self.Encoder2 = nn.Sequential(nn.Conv2d(n1, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(head_ch),
                                      nn.Sigmoid(), )

        # Decoder
        self.Decoder1 = nn.Sequential(nn.Conv2d(head_ch, n1, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU(), )

        self.Decoder2 = nn.Sequential(nn.Conv2d(n1, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(out_ch))

        # local branch, global branch, LRR branch
        self.branch1 = local_branch(in_ch=head_ch, num_module=local_blc)
        self.branch2 = Graph_branch(hidden_node=head_ch)
        self.LR_Prior = LRR_block(num_feat=head_ch, squeeze_factor=sq, memory_blocks=mb)
        self.CGAFusion = SFFM(dim=head_ch)
        self.Conv1x1 = nn.Conv2d(head_ch, head_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)

        local_br = self.branch1(x2)
        global_br = self.branch2(x2)
        LR_prior = self.LR_Prior(x2)
        fuse = self.CGAFusion(LR_prior, local_br, global_br)

        out1 = self.Decoder1(fuse)
        out2 = self.Decoder2(out1)

        return out2
