import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv


class gMLPLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, ffn_dim)
        self.sgu = SpatialGatingUnit(ffn_dim)
        self.proj_out = nn.Linear(ffn_dim, hidden_dim)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.proj_in.reset_parameters()
        self.sgu.reset_parameters()
        self.proj_out.reset_parameters()

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.sgu(x, edge_index)
        x = self.proj_out(x)
        return x


class SpatialGatingUnit(nn.Module):
    def __init__(self, ffn_dim, init_eps=1e-3):
        super().__init__()
        self.init_eps = init_eps / ffn_dim

        self.norm = nn.LayerNorm(ffn_dim)
        self.act = torch.tanh
        self.gcn = GCNConv(ffn_dim, ffn_dim, improved=True)

    def reset_parameters(self):
        self.norm.reset_parameters()
        nn.init.uniform_(self.gcn.lin.weight, -self.init_eps, self.init_eps)
        nn.init.constant_(self.gcn.bias, 1.0)

    def forward(self, x, edge_index):
        gate = self.norm(x)
        gate = self.gcn(gate, edge_index)
        gate = self.act(gate)
        return gate * x  # (batch, n_nodes, ffn_dim)
