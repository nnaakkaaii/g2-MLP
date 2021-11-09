import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.glob import global_sort_pool

from .modules import Residual


def create_network(num_features, num_classes, node_level, opt):
    return GAT(
        num_features=num_features,
        num_classes=num_classes,
        node_level=node_level,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        n_layers=opt.n_layers,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=128, help='中間層の特徴量')
    parser.add_argument('--n_heads', type=int, default=4, help='GATのhead数')
    parser.add_argument('--n_layers', type=int, default=3, help='GATの層数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class GAT(nn.Module):
    def __init__(self, num_features, num_classes, node_level, hidden_dim, n_heads, n_layers, dropout_rate, top_k=30):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.node_level = node_level
        self.top_k = top_k

        assert n_layers >= 2
        self.conv_layers = nn.ModuleList()
        self.conv_layers += [GATConv(num_features, hidden_dim, n_heads, concat=False, dropout=dropout_rate)]
        for _ in range(n_layers - 2):
            self.conv_layers += [Residual(GATConv(hidden_dim, hidden_dim, n_heads, concat=False, dropout=dropout_rate))]

        if node_level:
            self.conv_layers += [GATConv(hidden_dim, num_classes, n_heads, concat=False)]
        else:
            self.conv_layers += [GATConv(hidden_dim, 1, n_heads, concat=False, dropout=dropout_rate)]
            self.classifier_1 = nn.Linear(top_k, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()
        if hasattr(self, 'classifier_1'):
            self.classifier_1.reset_parameters()
        if hasattr(self, 'classifier_2'):
            self.classifier_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.conv_layers):
            if i > 0:
                x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
            x = layer(x, edge_index)

        if self.node_level:
            return x
        else:
            x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
            x = global_sort_pool(x, batch, k=self.top_k)
            x = self.classifier_1(x)
            x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
            x = self.classifier_2(x)
            return x
