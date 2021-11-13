import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

from .modules import Residual
from .utils.dropout_layers import dropout_layers


def create_network(num_features, num_classes, opt):
    return gMLP(
        num_features=num_features,
        num_classes=num_classes,
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


class gMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, prob_survival, top_k=30):
        super().__init__()
        self.prob_survival = prob_survival
        self.top_k = top_k

        assert n_layers >= 2
        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPBlock(hidden_dim, ffn_dim))]
        
        self.linear = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        prob_survival = self.prob_survival if self.training else 1

        x = self.embedding(x)
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x, edge_index)
        x = self.linear(x)

        return x


class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, ffn_dim)
        self.sgu = SpatialGatingUnit(ffn_dim)
        self.proj_out = nn.Linear(ffn_dim, hidden_dim)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.proj_in.reset_parameters()
        self.sgu.reset_parameters()
        self.proj_out.reset_parameters()

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.sgu(x, edge_index)
        x = self.proj_out(x)
        return x


class SpatialGatingUnit(nn.Module):
    def __init__(self, ffn_dim, init_eps=1e-3):
        super().__init__()
        self.init_eps = init_eps / ffn_dim

        self.norm = nn.LayerNorm(ffn_dim)
        self.act = torch.tanh
        self.gcn = GCNConv(ffn_dim, ffn_dim, improved=True)

    def reset_parameters(self):
        self.norm.reset_parameters()
        nn.init.uniform_(self.gcn.lin.weight, -self.init_eps, self.init_eps)
        nn.init.constant_(self.gcn.bias, 1.0)

    def forward(self, x, edge_index):
        gate = self.norm(x)
        gate = self.gcn(gate, edge_index)
        gate = self.act(gate)
        out = gate * x
        return out
