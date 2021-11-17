from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool

from .modules.gmlp_block import gMLPBlock


def create_network(num_features, num_classes, opt):
    return gMLPHierarchicalSAGPoolGraphClassification(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        n_hierarchies=opt.n_hierarchies,
        dropout_rate=opt.dropout_rate,
        pool_ratio=opt.pool_ratio,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=64, help='中間層の特徴量')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFNの特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='MLPの層数')
    parser.add_argument('--n_hierarchies', type=int, default=3, help='Poolingの層数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    parser.add_argument('--pool_ratio', type=float, default=0.25, help='1回あたりのプーリング率')
    return parser


class gMLPHierarchicalSAGPoolGraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__()
        assert n_layers >= 2
        assert n_hierarchies >= 1
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(n_hierarchies):
            self.layers += [gMLPBlock(hidden_dim, ffn_dim, n_layers)]
            self.pools += [SAGPooling(hidden_dim, pool_ratio)]

        self.linear1 = nn.Linear(2 * n_hierarchies * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.pools:
            layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        cat = []
        for layer, pool in zip(self.layers, self.pools):
            jk = []
            x, ys = layer(x, edge_index)
            jk += [F.gelu(torch.cat([global_mean_pool(y, batch), global_max_pool(y, batch)], dim=1)) for y in ys]
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)
            jk += [F.gelu(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))]
            cat += [sum(jk)]

        x = F.gelu(torch.cat(cat, dim=1))

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)
