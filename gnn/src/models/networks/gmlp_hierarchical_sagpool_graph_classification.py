from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool

from .modules.gmlp_block import gMLPBlock


def create_network(num_features, num_classes, opt):
    networks = {
        1: gMLPHierarchicalSAGPoolGraphClassification1,
        2: gMLPHierarchicalSAGPoolGraphClassification2,
        3: gMLPHierarchicalSAGPoolGraphClassification3,
        4: gMLPHierarchicalSAGPoolGraphClassification4,
        5: gMLPHierarchicalSAGPoolGraphClassification5,
        6: gMLPHierarchicalSAGPoolGraphClassification6,
        7: gMLPHierarchicalSAGPoolGraphClassification7,
    }
    return networks[opt.version](
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
    parser.add_argument('--version', type=int, default=1, help='バージョン. 1-7')
    return parser


class _gMLPHierarchicalSAGPoolGraphClassification(nn.Module):
    def __init__(self, num_classifier_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__()
        assert n_layers >= 2
        assert n_hierarchies >= 1
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(n_hierarchies - 1):
            self.layers += [gMLPBlock(hidden_dim, ffn_dim, n_layers)]
            self.pools += [SAGPooling(hidden_dim, pool_ratio)]
        self.layer = gMLPBlock(hidden_dim, ffn_dim, n_layers)

        self.linear1 = nn.Linear(num_classifier_dim, hidden_dim)
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

    def layer_readout(self, conv_outs, conv_batch):
        raise NotImplementedError

    def hierarchical_readout(self, layer_outs):
        raise NotImplementedError

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        layer_outs = []
        for layer, pool in zip(self.layers, self.pools):
            x, ys = layer(x, edge_index)
            layer_outs += [self.layer_readout(ys, batch)]
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)
        x, ys = self.layer(x, edge_index)
        layer_outs += [self.layer_readout(ys, batch)]

        x = self.hierarchical_readout(layer_outs)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)


class gMLPHierarchicalSAGPoolGraphClassification1(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [cat] -> [read_out] -> layer_readout
    layer_readouts -> [cat] -> [gelu] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * n_layers * n_hierarchies * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        x = torch.cat(conv_outs, dim=1)
        return torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)

    def hierarchical_readout(self, layer_outs):
        x = torch.cat(layer_outs, dim=1)
        return F.gelu(x)


class gMLPHierarchicalSAGPoolGraphClassification2(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [cat] -> [read_out] -> layer_readout
    layer_readouts -> [gelu] -> [sum] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * n_layers * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        x = torch.cat(conv_outs, dim=1)
        return torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)

    def hierarchical_readout(self, layer_outs):
        x = sum(map(F.gelu, layer_outs))
        return x


class gMLPHierarchicalSAGPoolGraphClassification3(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [-1] -> [read_out] -> layer_readout
    layer_readouts -> [cat] -> [gelu] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * n_hierarchies * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        x = conv_outs[-1]
        return torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)

    def hierarchical_readout(self, layer_outs):
        x = torch.cat(layer_outs, dim=1)
        return F.gelu(x)


class gMLPHierarchicalSAGPoolGraphClassification4(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [-1] -> [read_out] -> layer_readout
    layer_readouts -> [gelu] -> [sum] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        x = conv_outs[-1]
        return torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)

    def hierarchical_readout(self, layer_outs):
        x = sum(map(F.gelu, layer_outs))
        return x


class gMLPHierarchicalSAGPoolGraphClassification5(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [read_out] -> [gelu] -> [sum] -> layer_readout
    layer_readouts -> [cat] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * n_hierarchies * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        return sum(map(lambda x: F.gelu(torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)), conv_outs))

    def hierarchical_readout(self, layer_outs):
        x = torch.cat(layer_outs, dim=1)
        return x


class gMLPHierarchicalSAGPoolGraphClassification6(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [read_out] -> [gelu] -> [sum] -> layer_readout
    layer_readouts -> [sum] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        return sum(map(lambda x: F.gelu(torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)), conv_outs))

    def hierarchical_readout(self, layer_outs):
        x = sum(layer_outs)
        return x


class gMLPHierarchicalSAGPoolGraphClassification7(_gMLPHierarchicalSAGPoolGraphClassification):
    """
    conv_outs -> [-1] -> [read_out] -> layer_readout
    layer_readouts -> [-1] -> [gelu] -> classifier
    """
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio):
        super().__init__(2 * hidden_dim, num_features, num_classes, hidden_dim, ffn_dim, n_layers, n_hierarchies, dropout_rate, pool_ratio)

    def layer_readout(self, conv_outs, conv_batch):
        x = conv_outs[-1]
        return torch.cat((global_mean_pool(x, conv_batch), global_max_pool(x, conv_batch)), dim=1)

    def hierarchical_readout(self, layer_outs):
        x = layer_outs[-1]
        return F.gelu(x)
