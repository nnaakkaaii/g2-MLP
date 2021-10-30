import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_sort_pool

from .modules import GNN_TYPES, POOL_TYPES


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GNNSAGPool(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        gnn_type=opt.gnn_type,
        pool_type=opt.pool_type,
        hidden_dim=opt.hidden_dim,
        ratio=opt.ratio,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--gnn_type', type=str, choices=GNN_TYPES.keys(), help='利用するGNNのタイプ')
    parser.add_argument('--pool_type', type=str, choices=POOL_TYPES.keys(), help='利用するpoolingのタイプ')
    parser.add_argument('--hidden_dim', type=int, default=256, help='中間層の特徴量')
    parser.add_argument('--ratio', type=float, default=0.5, help='pooling率')
    parser.add_argument('--n_heads', type=int, default=4, help='並列数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GNNSAGPool(nn.Module):
    def __init__(self, num_features: int, num_classes: int, task_type: str, gnn_type: str, pool_type: str,
                 hidden_dim: int, ratio: float, n_heads: int, dropout_rate: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        assert task_type == 'graph_classification' 
 
        GNN = GNN_TYPES[gnn_type]
        Pool = POOL_TYPES[pool_type]

        gnn_kwargs = {}
        if gnn_type == 'GAT':
            gnn_kwargs['heads'] = n_heads
            gnn_kwargs['concat'] = False
            gnn_kwargs['dropout'] = dropout_rate
        elif gnn_type == 'GCN':
            gnn_kwargs['improved'] = True

        pool_kwargs = {}
        if pool_type == 'SAGPool':
            pool_kwargs['GNN'] = GNN_TYPES['GCN']
            pool_kwargs['improved'] = True

        self.conv1 = GNN(num_features, hidden_dim, **gnn_kwargs)
        self.pool1 = Pool(hidden_dim, ratio, **pool_kwargs)
        self.conv2 = GNN(hidden_dim, hidden_dim, **gnn_kwargs)
        self.pool2 = Pool(hidden_dim, ratio, **pool_kwargs)
        self.conv3 = GNN(hidden_dim * n_heads, 1, **gnn_kwargs)
        self.classifier_1 = nn.Linear(30, hidden_dim)
        self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool2.reset_parameters()
        self.conv3.reset_parameters()
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()
        return

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_1 = F.dropout(F.elu(self.conv1(x, edge_index), inplace=True))
        x_1, edge_index, edge_attr, _, _, _ = self.pool1(x_1, edge_index, edge_attr)
        x_2 = F.dropout(F.elu(self.conv2(x_1, edge_index), inplace=True))
        x_2, edge_index, edge_attr, batch, _, _ = self.pool2(x_2, edge_index, edge_attr)
        x_3 = self.conv3(x_2, edge_index)

        out = global_sort_pool(x_3, batch, k=30)
        out = F.elu(self.classifier_1(out), inplace=True)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.classifier_2(out)
        return out.view(1, -1)
