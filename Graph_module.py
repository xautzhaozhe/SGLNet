import torch
import torch.nn as nn


class GRAPHLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        # Construct a layer norm module in the TF style (epsilon inside the square root).
        super(GRAPHLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Graph_branch(nn.Module):
    def __init__(self, hidden_node):
        super(Graph_branch, self).__init__()
        self.node = hidden_node
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_node, out_channels=hidden_node, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_node),
            nn.LeakyReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_node, out_channels=hidden_node, kernel_size=1),
            nn.BatchNorm2d(hidden_node),
            nn.LeakyReLU())

        self.conv_out = nn.Conv2d(in_channels=hidden_node, out_channels=hidden_node, kernel_size=1)

        self.dropout = nn.Dropout(0.1)
        self.reps_graph = nn.Sequential(
            nn.Linear(hidden_node, int(hidden_node/2)),
            GRAPHLayerNorm(int(hidden_node/2), eps=1e-12),
            nn.ReLU(),
            nn.Linear(int(hidden_node / 2), hidden_node),
            GRAPHLayerNorm(hidden_node, eps=1e-12),
            nn.ReLU(),
        )

        self.Down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='nearest')
        self.hidden = hidden_node

    def forward(self, x):
        org = x
        x = self.Down(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(1, H*W, C)

        # Construct matrix
        reps_graph = torch.matmul(x, x.permute(0, 2, 1))
        reps_graph = nn.Softmax(dim=-1)(reps_graph)
        rel_reps = torch.matmul(reps_graph, x)
        rel_reps = self.reps_graph(x + rel_reps)

        # Reshape
        rel_reps = rel_reps.reshape(1, H, W, C)
        rel_reps1 = rel_reps.permute(0, 3, 1, 2)

        # Up sample
        rel_reps1 = self.Up(rel_reps1)
        rel_reps1 = self.conv2(rel_reps1)
        out = self.conv_out(rel_reps1 + org)

        return out





