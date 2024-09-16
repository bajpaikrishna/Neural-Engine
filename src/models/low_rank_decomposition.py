import torch
import torch.nn as nn

class LowRankDecomposition(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LowRankDecomposition, self).__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x):
        return torch.matmul(x, torch.matmul(self.U, self.V))
