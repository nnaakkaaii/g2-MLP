import argparse

import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

from .base_3layer_gnn import Base3LayerGNN


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GCN(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        hidden_dim=opt.hidden_dim,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--hidden_dim', type=int, default=256, help='中間層の特徴量')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GCN(Base3LayerGNN):
    def __init__(self, num_features: int, num_classes: int, task_type: str, hidden_dim: int, dropout_rate: float):
        super().__init__(task_type, dropout_rate)
        self.num_classes = num_classes

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        if task_type == 'multi_label_node_classification':
            self.conv3 = GCNConv(hidden_dim, 2 * num_classes)
        elif task_type in ['node_classification', 'node_regression']:
            self.conv3 = GCNConv(hidden_dim, num_classes)
        elif task_type == 'graph_classification':
            self.conv3 = GCNConv(hidden_dim, 1)
            self.classifier_1 = nn.Linear(30, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)
        else:
            raise NotImplementedError

        self.reset_parameters()
