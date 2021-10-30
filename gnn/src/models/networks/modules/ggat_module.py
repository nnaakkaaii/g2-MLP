import abc
from typing import Callable, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.typing import Adj


class AbstractGGATBlock(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, in_features: int, out_features: int, GNN: Callable,
                 dropout_rate: float, nonlinearity: Callable, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.nonlinearity = nonlinearity

    @abc.abstractmethod
    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, edge_index: Adj, y: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor
        edge_index : Adj
            adjacency matrix of input tensor
        y : Optional[torch.Tensor], default None
            gate tensor

        Returns
        -------
        x : torch.Tensor
            output tensor
        score : torch.Tensor
            attention score
        """
        pass


class GGAT1Block(AbstractGGATBlock):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.tanh, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate, nonlinearity, **kwargs)
        self.gnn_1 = GNN(in_features, 1, **kwargs)  # heads, concat, dropoutのkwargsが存在
        self.gnn_2 = GNN(in_features, out_features, **kwargs)

    def reset_parameters(self):
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj, y: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if y is None:
            x_1 = x_2 = x
        else:
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            assert x.shape == y.shape, 'GGATへの入力次元が異なります'
            x_1 = x + y
            x_2 = x
        
        x_1 = self.nonlinearity(self.gnn_1(x_1, edge_index))  # ノード数 x 1
        x_2 = self.gnn_2(x_2, edge_index)  # ノード数 x out_features

        x = x_1 * x_2
        x = F.dropout(F.elu(x, inplace=True), p=self.dropout_rate, training=self.training)

        return x, x_1.view(-1)


class GGAT2Block(AbstractGGATBlock):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv,
                 dropout_rate: float = 0, nonlinearity: Callable = torch.softmax, **kwargs):
        super().__init__(in_features, out_features, GNN, dropout_rate, nonlinearity, **kwargs)
        self.gnn_0 = GNN(in_features, out_features, **kwargs)  # heads, concat, dropoutのkwargsが存在
        self.gnn_1 = GNN(in_features, out_features, **kwargs)
        self.gnn_2 = GNN(in_features, out_features, **kwargs)

    def reset_parameters(self):
        self.gnn_0.reset_parameters()
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj, y: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if y is None:
            x_0 = x_1 = x_2 = x
        else:
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            x_0 = y
            x_1 = x_2 = x
        
        x_0 = self.gnn_0(x_0, edge_index)
        x_1 = self.gnn_1(x_1, edge_index)

        x_1 = self.nonlinearity(torch.mm(x_1, x_0.T)) / x_1.size(-1) ** 0.5
        x_2 = self.gnn_2(x_2, edge_index)

        x = torch.mm(x_1, x_2)
        x = F.dropout(F.elu(x, inplace=True), p=self.dropout_rate, training=self.training)

        return x, x_1.view(-1)


class GGATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, GNN: Callable = GraphConv, dropout_rate: float = 0,
                 nonlinearity: Callable = torch.tanh, skip_connection: bool = True, GGATBlock: AbstractGGATBlock = GGAT1Block,
                 ggat_heads: int = 1, ratio: Union[float, int] = 1, ggat_concat: bool = False, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.ggat_concat = ggat_concat
        self.skip_connection = skip_connection

        self.heads = nn.ModuleList()
        for _ in range(ggat_heads):
            self.heads.append(GGATBlock(
                in_features, out_features, GNN=GNN,
                dropout_rate=dropout_rate, nonlinearity=nonlinearity, **kwargs
            ))

    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        outs = [head(x, edge_index, y) for head in self.heads]
        if self.ggat_concat:
            x_all = torch.cat([out[0] for out in outs], dim=-1)
        else:
            x_all = sum([out[0] for out in outs]) / len(outs)
        
        if self.skip_connection:
            x_all = x_all + x

        score = sum([out[1] for out in outs]) / len(outs)

        if self.ratio < 1 and edge_attr is None:
            warn('poolingを利用する場合は、edge_attrを入力してください')
        if self.ratio < 1 and edge_attr is not None:
            perm = topk(score, self.ratio, batch)

            x_all = x_all[perm]
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

            return x_all, edge_index, edge_attr, batch, perm, score[perm]
        else:
            return x_all
