import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .utils.dropout_layers import dropout_layers
from .modules.residual import Residual
from .modules.gmlp_layer import gMLPLayer


def create_network(num_features, num_classes, opt):
    return gMLPGraphClassification(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=opt.hidden_dim,
        ffn_dim=opt.ffn_dim,
        n_layers=opt.n_layers,
        prob_survival=opt.prob_survival,
        dropout_rate=opt.dropout_rate,
    )


def network_modify_commandline_options(parser):
    parser.add_argument('--hidden_dim', type=int, default=64, help='中間層の特徴量')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFNの特徴量')
    parser.add_argument('--n_layers', type=int, default=3, help='MLPの層数')
    parser.add_argument('--prob_survival', type=float, default=1., help='layerのdropout率')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropoutの割合')
    return parser


class gMLPGraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, prob_survival, dropout_rate):
        super().__init__()
        assert n_layers >= 2
        self.prob_survival = prob_survival
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPLayer(hidden_dim, ffn_dim))]

        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        prob_survival = self.prob_survival if self.training else 1
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        jk = F.gelu(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x, edge_index)
            jk += F.gelu(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))

        x = F.relu(self.linear1(jk))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)
