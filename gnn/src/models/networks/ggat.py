import argparse

import torch.nn as nn

from .base_3layer_gnn import Base3LayerGNN
from .modules import GGAT_TYPES, GNN_TYPES
from .modules.ggat_module import GGATLayer


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GGAT(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        ggat_type=opt.ggat_type,
        ggat_heads=opt.ggat_heads,
        gnn_type=opt.gnn_type,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ggat_type', type=str, choices=GGAT_TYPES.keys(), help='利用するGGATのタイプ')
    parser.add_argument('--ggat_heads', type=int, default=4, help='GGATの並列数')
    parser.add_argument('--gnn_type', type=str, choices=GNN_TYPES.keys(), help='利用するGNNのタイプ')
    parser.add_argument('--hidden_dim', type=int, default=256, help='中間層の特徴量')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    parser.add_argument('--n_heads', type=int, default=4, help='GATの並列数(GNNがGATの時のみ有効)')
    return parser


class GGAT(Base3LayerGNN):
    def __init__(self, num_features: int, num_classes: int, task_type: str, ggat_type: str, ggat_heads: int,
                 gnn_type: str, hidden_dim: int, n_heads: int, dropout_rate: float):
        super().__init__(task_type, dropout_rate)
        self.num_classes = num_classes

        GGAT = GGAT_TYPES[ggat_type]
        GNN = GNN_TYPES[gnn_type]

        kwargs = {}
        if gnn_type == 'GAT':
            kwargs['heads'] = n_heads
            kwargs['concat'] = False
            kwargs['dropout'] = dropout_rate

        self.conv1 = GGATLayer(num_features, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                               skip_connection=False, GGATBlock=GGAT,
                               ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        self.conv2 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                               skip_connection=True, GGATBlock=GGAT,
                               ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        if task_type == 'multi_label_node_classification':
            self.conv3 = GGATLayer(hidden_dim * ggat_heads, 2 * num_classes, GNN=GNN, dropout_rate=dropout_rate,
                                   skip_connection=False, GGATBlock=GGAT,
                                   ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
        elif task_type in ['node_classification', 'node_regression']:
            self.conv3 = GGATLayer(hidden_dim * ggat_heads, num_classes, GNN=GNN, dropout_rate=dropout_rate,
                                   skip_connection=False, GGATBlock=GGAT,
                                   ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
        elif task_type == 'graph_classification':
            self.conv3 = GGATLayer(hidden_dim * ggat_heads, 1, GNN=GNN, dropout_rate=dropout_rate,
                                   skip_connection=False, GGATBlock=GGAT,
                                   ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
            self.classifier_1 = nn.Linear(30, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)
        else:
            raise NotImplementedError

        self.reset_parameters()
