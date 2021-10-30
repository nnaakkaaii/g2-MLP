import abc
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class AbstractGGATLayer(nn.Module, meta=abc.ABCMeta):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = True, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.ratio = ratio
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity

    @abc.abstractmethod
    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        pass


class AbstractGGAT(nn.Module, meta=abc.ABCMeta):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = False,
                 ggat_heads: int = 1, concat: bool = False, **kwargs):
        super().__init__()
        self.concat = concat

    @abc.abstractmethod
    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        pass


class GGAT1Layer(AbstractGGATLayer):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = True, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate,
                         nonlinearity, ratio, skip_connection, **kwargs)
        self.gnn_1 = GNN(in_features, 1, **kwargs)  # TODO: n_heads, concat, dropout を考慮できていない
        self.gnn_2 = GNN(in_features, out_features, **kwargs)

    def reset_parameters(self):
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_1 = x.unsqueeze(-1) if x.dim() == 1 else x
        if y is None:
            x_2 = x_1
        else:
            y_1 = y.unsqueeze(-1) if y.dim() == 1 else y
            assert x_1.shape == y_1.shape, 'GGATへの入力次元が異なります'
            x_2 = x_1 + y_1
        score = self.gnn_1(x_2, edge_index).view(-1)
        score = self.nonlinearity(score)
        perm = topk(score, self.ratio, batch)

        x_3 = self.gnn_2(x_1, edge_index)
        x_4 = F.dropout(x_3, p=self.dropout_rate, training=self.training)
        if not self.skip_connection:
            x_5 = x_4[perm] * score[perm].view(-1, 1)
        else:
            x_5 = x_1[perm] + x_4[perm] * score[perm].view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x_5, edge_index, edge_attr, batch, perm, score[perm]


class GGAT1(AbstractGGAT):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = False,
                 ggat_heads: int = 1, concat: bool = False, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate, nonlinearity,
                         ratio, skip_connection, ggat_heads, concat, **kwargs)

        self.layers = nn.ModuleList()
        for _ in range(ggat_heads):
            self.layers.append(GGAT1Layer(in_features, out_features, GNN,
                                          dropout_rate, nonlinearity,
                                          ratio, skip_connection, **kwargs))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return

    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        if self.concat:
            return torch.cat([layer(x, edge_index, edge_attr, batch, y) for layer in self.layers], dim=-1)
        else:
            return torch.mean([layer(x, edge_index, edge_attr, batch, y) for layer in self.layers], dim=-1)[0]


class GGAT2Layer(AbstractGGATLayer):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = True, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate,
                         nonlinearity, ratio, skip_connection, **kwargs)
        self.gnn_1 = GNN(in_features, out_features, **kwargs)
        self.gnn_2 = GNN(in_features, out_features, **kwargs)
        self.gnn_3 = GNN(in_features, out_features, **kwargs)

    def reset_parameters(self):
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_1 = x.unsqueeze(-1) if x.dim() == 1 else x
        if y is None:
            y_1 = x_1
        else:
            y_1 = y.unsqueeze(-1) if y.dim() == 1 else y
        x_2 = self.gnn_1(x_1, edge_index).view(-1)
        y_2 = self.gnn_2(y_1, edge_index).view(-1)
        score = torch.bmm(x_2, y_2.T) / x_2.size(-1) ** 0.5
        perm = topk(score, self.ratio, batch)

        x_3 = self.gnn_3(x_1, edge_index).view(-1)
        x_4 = F.dropout(x_3, p=self.dropout_rate, training=self.training)

        if not self.skip_connection:
            x_5 = torch.bmm(score[perm].view(-1, 1), x_4[perm])
        else:
            x_5 = x_1[perm] +torch.bmm(score[perm].view(-1, 1), x_4[perm])

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x_5, edge_index, edge_attr, batch, perm, score[perm]


class GGAT2(AbstractGGAT):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh,
                 ratio: Union[float, int] = 1, skip_connection: bool = False,
                 ggat_heads: int = 1, concat: bool = False, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate, nonlinearity,
                         ratio, skip_connection, ggat_heads, concat, **kwargs)

        self.layers = nn.ModuleList()
        for _ in range(ggat_heads):
            self.layers.append(GGAT2Layer(in_features, out_features, GNN,
                                          dropout_rate, nonlinearity,
                                          ratio, skip_connection, **kwargs))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return

    def forward(self, x, edge_index, edge_attr, batch=None, y=None):
        if self.concat:
            return torch.cat([layer(x, edge_index, edge_attr, batch, y) for layer in self.layers], dim=-1)
        else:
            return torch.mean([layer(x, edge_index, edge_attr, batch, y) for layer in self.layers], dim=-1)[0]
