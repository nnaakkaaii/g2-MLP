import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

from .modules import Residual
from .utils.dropout_layers import dropout_layers


def create_network(num_features, num_classes, is_graph_classification, opt):
    return MLP(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        prob_survival=opt.prob_survival,
        dropout_rate=opt.dropout_rate,
        is_graph_classification=is_graph_classification,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=64, help='中間層の特徴量')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFNの特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='MLPの層数')
    parser.add_argument('--prob_survival', type=float, default=1., help='layerのdropout率')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout率')
    return parser


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, prob_survival, dropout_rate, is_graph_classification):
        super().__init__()
        self.prob_survival = prob_survival
        self.dropout_rate = dropout_rate

        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(num_features, hidden_dim)]
        for _ in range(n_layers - 2):
            self.layers += [Residual(MLPBlock(hidden_dim, ffn_dim))]
        
        if not is_graph_classification:
            self.layers += [nn.Linear(hidden_dim, num_classes)]
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self, 'classifiers'):
            for layer in self.classifiers:
                layer.reset_parameters()

    def forward(self, data):
        x, batch = data.x, data.batch

        prob_survival = self.prob_survival if self.training else 1
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x)

        if hasattr(self, 'classifiers'):
            x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            for i, layer in enumerate(self.classifiers):
                if i > 0:
                    x = F.dropout(F.gelu(x), p=self.dropout_rate, training=self.training)
                x = layer(x)
        
        return x


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, ffn_dim)
        self.proj_out = nn.Linear(ffn_dim, hidden_dim)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.proj_in.reset_parameters()
        self.proj_out.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.proj_out(x)
        return x
