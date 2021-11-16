from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool

from .modules.residual import Residual
from .modules.gmlp_layer import gMLPLayer


def create_network2(num_features, num_classes, opt):
    return gMLPSAGPoolGraphClassification2(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        n_hierarchies=opt.n_hierarchies,
        dropout_rate=opt.dropout_rate,
        pool_ratio=opt.pool_ratio,
        jk=opt.jk,
    )


def create_network3(num_features, num_classes, opt):
    return gMLPSAGPoolGraphClassification3(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        n_hierarchies=opt.n_hierarchies,
        dropout_rate=opt.dropout_rate,
        pool_ratio=opt.pool_ratio,
        jk=opt.jk,
    )


def create_network4(num_features, num_classes, opt):
    return gMLPSAGPoolGraphClassification4(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        n_hierarchies=opt.n_hierarchies,
        dropout_rate=opt.dropout_rate,
        pool_ratio=opt.pool_ratio,
        jk=opt.jk,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=64, help='中間層の特徴量')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFNの特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='MLPの層数')
    parser.add_argument('--n_hierarchies', type=int, default=3, help='Poolingの層数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    parser.add_argument('--pool_ratio', type=float, default=0.25, help='1回あたりのプーリング率')
    parser.add_argument('--jk', action='store_true', help='jump knowledgeの追加')
    return parser


class gMLPSAGPoolGraphClassification2(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio, jk=True):
        super().__init__()
        assert n_layers >= 2
        assert n_hierarchies >= 1
        self.jk = jk
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(n_hierarchies - 1):
            self.layers += [gMLPBlock(hidden_dim, ffn_dim, n_layers)]
            self.pools += [SAGPooling(hidden_dim, pool_ratio)]

        self.layer = gMLPBlock(hidden_dim, ffn_dim, n_layers)

        self.linear1 = nn.Linear(2 * n_layers * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.pools:
            layer.reset_parameters()
        self.layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        jk = []
        for layer, pool in zip(self.layers, self.pools):
            x, ys = layer(x, edge_index)
            # gMLPBlockの各出力を横並びにしてmean-max
            jk += [F.gelu(torch.cat([global_mean_pool(y, batch) for y in ys] + [global_max_pool(y, batch) for y in ys], dim=1))]
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)

        _, ys = self.layer(x, edge_index)
        jk += [F.gelu(torch.cat([global_mean_pool(y, batch) for y in ys] + [global_max_pool(y, batch) for y in ys], dim=1))]

        x = sum(jk)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)


class gMLPSAGPoolGraphClassification3(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio, jk=True):
        super().__init__()
        assert n_layers >= 2
        assert n_hierarchies >= 1
        self.jk = jk
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(n_hierarchies - 1):
            self.layers += [gMLPBlock(hidden_dim, ffn_dim, n_layers)]
            self.pools += [SAGPooling(hidden_dim, pool_ratio)]

        self.layer = gMLPBlock(hidden_dim, ffn_dim, n_layers)

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
        self.layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        jk = []
        for layer, pool in zip(self.layers, self.pools):
            x, ys = layer(x, edge_index)
            jk += [sum(F.gelu(torch.cat([global_mean_pool(y, batch), global_max_pool(y, batch)], dim=1)) for y in ys)]
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)

        _, ys = self.layer(x, edge_index)
        jk += [sum(F.gelu(torch.cat([global_mean_pool(y, batch), global_max_pool(y, batch)], dim=1)) for y in ys)]

        x = torch.cat(jk, dim=1)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)


class gMLPSAGPoolGraphClassification4(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio, jk=True):
        super().__init__()
        assert n_layers >= 2
        assert n_hierarchies >= 1
        self.jk = jk
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(n_hierarchies - 1):
            self.layers += [gMLPBlock(hidden_dim, ffn_dim, n_layers)]
            self.pools += [SAGPooling(hidden_dim, pool_ratio)]

        self.layer = gMLPBlock(hidden_dim, ffn_dim, n_layers)

        self.linear1 = nn.Linear(2 * n_hierarchies * n_layers * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.pools:
            layer.reset_parameters()
        self.layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        jk = []
        for layer, pool in zip(self.layers, self.pools):
            x, ys = layer(x, edge_index)
            jk += [F.gelu(torch.cat([global_mean_pool(y, batch) for y in ys] + [global_max_pool(y, batch) for y in ys], dim=1))]
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)

        _, ys = self.layer(x, edge_index)
        jk += [F.gelu(torch.cat([global_mean_pool(y, batch) for y in ys] + [global_max_pool(y, batch) for y in ys], dim=1))]

        x = torch.cat(jk, dim=1)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)


class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_layers):
        super().__init__()
        assert n_layers >= 2

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPLayer(hidden_dim, ffn_dim))]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self, 'linear'):
            self.linear.reset_parameters()
    
    def forward(self, x, edge_index):
        
        xs = []
        for layer in self.layers:
            x = layer(x, edge_index)
            xs += [x]

        return x, xs
