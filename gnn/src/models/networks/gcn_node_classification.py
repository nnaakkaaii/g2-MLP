import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

from .modules import Residual


def create_network(num_features, num_classes, opt):
    return GCNNodeClassification(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        n_layers=opt.n_layers,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=128, help='中間層の特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='GCNの層数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GCNNodeClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

        assert n_layers >= 2
        self.conv_layers = nn.ModuleList()
        self.conv_layers += [GCNConv(num_features, hidden_dim, improved=True)]
        for _ in range(n_layers - 2):
            self.conv_layers += [Residual(GCNConv(hidden_dim, hidden_dim, improved=True))]

        self.conv_layers += [GCNConv(hidden_dim, num_classes, improved=True)]

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.conv_layers):
            if i > 0:
                x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
            x = layer(x, edge_index)

        return x
