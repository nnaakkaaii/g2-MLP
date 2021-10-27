import argparse

import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATConv


def create_network(num_features: int, num_classes: int, opt: argparse.Namespace) -> nn.Module:
    return GAT(
        num_features=num_features,
        num_classes=num_classes,
        is_regression=opt.is_regression,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--hidden_dim', type=int, default=256, help='中間層の特徴量')
    parser.add_argument('--n_heads', type=int, default=4, help='並列数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GAT(nn.Module):
    def __init__(self, num_features: int, num_classes: int, is_regression: bool, hidden_dim: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.is_regression = is_regression

        self.conv1 = GATConv(num_features, hidden_dim, heads=n_heads, concat=True, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True, dropout=dropout_rate)
        if is_regression:
            self.conv3 = GATConv(hidden_dim * n_heads, num_classes, heads=n_heads, concat=False, dropout=dropout_rate)
        else:
            self.conv3 = GATConv(hidden_dim * n_heads, 2 * num_classes, heads=n_heads, concat=False, dropout=dropout_rate)

        self.elu = nn.ELU(inplace=True)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x_1 = self.elu(self.conv1(x, edge_index))
        x_2 = self.elu(self.conv2(x_1, edge_index) + x_1)
        out = self.conv3(x_2, edge_index)
        if self.is_regression:
            out = out.view(-1)  # ノード x 特徴量 -> flatten
        else:
            out = out.view(-1, 2)

        return out
