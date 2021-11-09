from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.glob import global_sort_pool

from .modules import Residual
from .utils.dropout_layers import dropout_layers


def create_network(num_features, num_classes, node_level, opt):
    return MLP(
        num_features=num_features,
        num_classes=num_classes,
        node_level=node_level,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        prob_survival=opt.prob_survival,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=64, help='中間層の特徴量')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFNの特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='MLPの層数')
    parser.add_argument('--prob_survival', type=float, default=1., help='layerのdropout率')
    return parser


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, node_level, hidden_dim, ffn_dim, n_layers, prob_survival, top_k=30):
        super().__init__()
        self.node_level = node_level
        self.prob_survival = prob_survival
        self.top_k = top_k

        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(num_features, hidden_dim)]
        for _ in range(n_layers - 2):
            self.layers += [Residual(MLPBlock(hidden_dim, ffn_dim))]
        
        if node_level:
            self.layers += [nn.Linear(hidden_dim, num_classes)]
        else:
            self.layers += [nn.Linear(hidden_dim, 1)]
            self.classifier_1 = nn.Linear(top_k, hidden_dim)
            self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self, 'classifier_1'):
            self.classifier_1.reset_parameters()
        if hasattr(self, 'classifier_2'):
            self.classifier_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        prob_survival = self.prob_survival if self.training else 1
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x)
        
        if self.node_level:
            return x
        else:
            x = F.gelu(x)
            x = global_sort_pool(x, batch, k=self.top_k)
            x = self.classifier_1(x)
            x = F.gelu(x)
            x = self.classifier_2(x)
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
