import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_sort_pool

from .modules import GGAT_TYPES, GNN_TYPES
from .modules.ggat_module import GGATLayer
from .utils import augment_adj


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GGATUNet(
        num_features=num_features,
        num_classes=num_classes,
        task_type=opt.task_type,
        ggat_type=opt.ggat_type,
        ggat_heads=opt.ggat_heads,
        gnn_type=opt.gnn_type,
        hidden_dim=opt.hidden_dim,
        ratio=opt.ratio,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ggat_type', type=str, choices=GGAT_TYPES.keys(), help='利用するGGATのタイプ')
    parser.add_argument('--ggat_heads', type=int, default=2, help='並列数')
    parser.add_argument('--gnn_type', type=str, choices=GNN_TYPES.keys(), help='利用するGNNのタイプ')
    parser.add_argument('--hidden_dim', type=int, default=128, help='中間層の特徴量')
    parser.add_argument('--ratio', type=float, default=0.5, help='pooling率')
    parser.add_argument('--n_heads', type=int, default=4, help='並列数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GGATUNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int, task_type: str, ggat_type: str, ggat_heads: int,
                 gnn_type: str, hidden_dim: int, ratio: float, n_heads: int, dropout_rate: float):
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout_rate

        GGAT = GGAT_TYPES[ggat_type]
        GNN = GNN_TYPES[gnn_type]

        kwargs = {}
        if gnn_type == 'GAT':
            kwargs['heads'] = n_heads
            kwargs['concat'] = False
            kwargs['dropout'] = dropout_rate

        self.conv = GGATLayer(num_features, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                              skip_connection=False, GGATBlock=GGAT, ratio=1,
                              ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        self.down_conv1 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                                    skip_connection=True, GGATBlock=GGAT, ratio=ratio,
                                    ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        self.down_conv2 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                                    skip_connection=True, GGATBlock=GGAT, ratio=ratio,
                                    ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        self.down_conv3 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                                    skip_connection=True, GGATBlock=GGAT, ratio=ratio,
                                    ggat_heads=ggat_heads, ggat_concat=True, **kwargs)

        self.up_conv1 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                                  skip_connection=True, GGATBlock=GGAT, ratio=1,
                                  ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        self.up_conv2 = GGATLayer(hidden_dim * ggat_heads, hidden_dim, GNN=GNN, dropout_rate=dropout_rate,
                                  skip_connection=True, GGATBlock=GGAT, ratio=1,
                                  ggat_heads=ggat_heads, ggat_concat=True, **kwargs)
        if task_type == 'multi_label_node_classification':
            self.up_conv3 = GGATLayer(hidden_dim * ggat_heads, 2 * num_classes, GNN=GNN, dropout_rate=dropout_rate,
                                      skip_connection=False, GGATBlock=GGAT, ratio=1,
                                      ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
        elif task_type in ['node_classification', 'node_regression']:
            self.up_conv3 = GGATLayer(hidden_dim * ggat_heads, num_classes, GNN=GNN, dropout_rate=dropout_rate,
                                      skip_connection=False, GGATBlock=GGAT, ratio=1,
                                      ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
        elif task_type in ['graph_classification']:
            self.up_conv3 = GGATLayer(hidden_dim * ggat_heads, 1, GNN=GNN, dropout_rate=dropout_rate,
                                      skip_connection=False, GGATBlock=GGAT, ratio=1,
                                      ggat_heads=ggat_heads, ggat_concat=False, **kwargs)
            self.classifier_1 = nn.Linear(30, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.down_conv1.reset_parameters()
        self.down_conv2.reset_parameters()
        self.down_conv3.reset_parameters()
        self.up_conv1.reset_parameters()
        self.up_conv2.reset_parameters()
        self.up_conv3.reset_parameters()
        if hasattr(self, 'classifier_1'):
            self.classifier_1.reset_parameters()
        if hasattr(self, 'classifier_2'):
            self.classifier_2.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x1 = self.conv(x, edge_index)
        x1 = F.dropout(F.elu(x1, inplace=True), p=self.dropout_rate, training=self.training)
        edge_index1 = edge_index
        edge_weight1 = edge_weight

        edge_index2, edge_weight2 = augment_adj(edge_index1, edge_weight1, x1.size(0))
        x2, edge_index2, edge_weight2, batch, perm1, _ = self.down_conv1(x1, edge_index2, edge_weight2, batch)
        x2 = F.dropout(F.elu(x2, inplace=True), p=self.dropout_rate, training=self.training)

        edge_index3, edge_weight3 = augment_adj(edge_index2, edge_weight2, x2.size(0))
        x3, edge_index3, edge_weight3, batch, perm2, _ = self.down_conv2(x2, edge_index3, edge_weight3, batch)
        x3 = F.dropout(F.elu(x3, inplace=True), p=self.dropout_rate, training=self.training)

        edge_index4, edge_weight4 = augment_adj(edge_index3, edge_weight3, x3.size(0))
        x4, edge_index4, edge_weight4, batch, perm3, _ = self.down_conv3(x3, edge_index4, edge_weight4, batch)
        x4 = F.dropout(F.elu(x4, inplace=True), p=self.dropout_rate, training=self.training)

        up3 = torch.zeros_like(x3)
        up3[perm3] = x4
        x3 = self.up_conv1(up3, edge_index3, y=x3)
        x3 = F.dropout(F.elu(x3, inplace=True), p=self.dropout_rate, training=self.training)

        up2 = torch.zeros_like(x2)
        up2[perm2] = x3
        x2 = self.up_conv2(up2, edge_index2, y=x2)
        x2 = F.dropout(F.elu(x2, inplace=True), p=self.dropout_rate, training=self.training)

        up1 = torch.zeros_like(x1)
        up1[perm1] = x2
        x1 = self.up_conv3(up1, edge_index1, y=x1)

        if self.task_type == 'node_regression':
            return x1.view(-1)
        if self.task_type == 'multi_label_node_classification':
            return x1.view(-1, 2)
        if self.task_type == 'node_classification':
            return x1
        if self.task_type == 'graph_classification':
            out = global_sort_pool(x1, batch, k=30)
            out = F.elu(self.classifier_1(out), inplace=True)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            out = self.classifier_2(out)
            return out.view(1, -1)
        
        raise NotImplementedError
