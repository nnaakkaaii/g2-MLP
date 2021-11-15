import torch.nn as nn

from .utils.dropout_layers import dropout_layers
from .modules.residual import Residual
from .modules.gmlp_layer import gMLPLayer


def create_network(num_features, num_classes, opt):
    return gMLPNodeClassification(
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


class gMLPNodeClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, prob_survival):
        super().__init__()
        self.prob_survival = prob_survival

        assert n_layers >= 2
        self.embedding = nn.Linear(num_features, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPLayer(hidden_dim, ffn_dim))]

        self.classifier = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.classifier:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        prob_survival = self.prob_survival if self.training else 1

        x = self.embedding(x)
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x, edge_index)
        for layer in self.classifier:
            x = layer(x)

        return x
