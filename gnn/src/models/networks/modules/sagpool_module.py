from typing import Union, Callable

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class SAGPooling(nn.Module):
    def __init__(self, in_features: int, GNN: Callable = GraphConv, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 0.5, **kwargs):
        super().__init__()
        self.ratio = ratio

        self.gnn = GNN(in_features, 1, **kwargs)
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.gnn(x, edge_index).view(-1)
        score = self.nonlinearity(score)

        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]
