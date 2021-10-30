"""
import argparse
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn import global_sort_pool

from .modules.ggat_module import AbstractGGAT, GGAT1, GGAT2

GGAT_TYPES: Dict[str, AbstractGGAT] = {
    'GGAT1': GGAT1,
    'GGAT2': GGAT2,
}

GNN_TYPES: Dict[str, nn.Module] = {
    'GCN': GCNConv,
    'GAT': GATConv,
}


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GGATPool(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        ggat_type=opt.ggat_type,
        gnn_type=opt.gnn_type,
        hidden_dim=opt.hidden_dim,
        ratio=opt.ratio,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ggat_type', type=str, choices=GGAT_TYPES.keys(), help='利用するGGATのタイプ')
    parser.add_argument('--gnn_type', type=str, choices=GNN_TYPES.keys(), help='利用するGNNのタイプ')
    parser.add_argument('--hidden_dim', type=int, default=256, help='中間層の特徴量')
    parser.add_argument('--ratio', type=float, default=0.5, help='pooling率')
    parser.add_argument('--n_heads', type=int, default=4, help='並列数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GGATPool(nn.Module):
    def __init__(self, num_features: int, num_classes: int, task_type: str, ggat_type: str,
                 gnn_type: str, hidden_dim: int, ratio: float, n_heads: int, dropout_rate: float):
        super().__init__(task_type, dropout_rate)
        self.num_classes = num_classes

        assert task_type == 'graph_classification'    
        GGAT = GGAT_TYPES[ggat_type]
        GNN = GNN_TYPES[gnn_type]
        self.conv1 = GGAT(num_features, hidden_dim, GNN=GNN, ratio=ratio,
                          dropout_rate=dropout_rate, ggat_heads=n_heads, concat=True)
        self.conv2 = GGAT(hidden_dim * n_heads, hidden_dim, GNN=GNN, ratio=ratio, skip_connection=True,
                          dropout_rate=dropout_rate, ggat_heads=n_heads, concat=True)
        self.conv3 = GGAT(hidden_dim * n_heads, 1, GNN=GNN, ratio=1, skip_connection=True,
                            dropout_rate=dropout_rate, ggat_heads=n_heads, concat=False)
        self.classifier_1 = nn.Linear(30, hidden_dim)
        self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()
        return

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_1 = F.dropout(F.elu(self.conv1(x, edge_index), inplace=True))
        x_2 = F.dropout(F.elu(self.conv2(x_1, edge_index), inplace=True))
        x_3 = self.conv3(x_2, edge_index)

        out = global_sort_pool(x_3, batch, k=30)
        out = F.elu(self.classifier_1(out), inplace=True)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.classifier_2(out)
        return out.view(1, -1)
"""