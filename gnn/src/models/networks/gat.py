import argparse

import torch.nn as nn
from torch_geometric.nn.conv import GATConv

from .base_3layer_gnn import Base3LayerGNN


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GAT(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--hidden_dim', type=int, default=128, help='中間層の特徴量')
    parser.add_argument('--n_heads', type=int, default=4, help='並列数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GAT(Base3LayerGNN):
    def __init__(self, num_features: int, num_classes: int, task_type: str, hidden_dim: int, n_heads: int, dropout_rate: float):
        super().__init__(task_type, dropout_rate)
        self.num_classes = num_classes

        self.conv1 = GATConv(num_features, hidden_dim, heads=n_heads, concat=True, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True, dropout=dropout_rate)
        if task_type == 'multi_label_node_classification':
            self.conv3 = GATConv(hidden_dim * n_heads, 2 * num_classes, heads=n_heads, concat=False, dropout=dropout_rate)
        elif task_type in ['node_classification', 'node_regression']:
            self.conv3 = GATConv(hidden_dim * n_heads, num_classes, heads=n_heads, concat=False, dropout=dropout_rate)
        elif task_type == 'graph_classification':
            self.conv3 = GATConv(hidden_dim * n_heads, 1, heads=n_heads, concat=False, dropout=dropout_rate)
            self.classifier_1 = nn.Linear(30, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)
        else:
            raise NotImplementedError

        self.reset_parameters()
