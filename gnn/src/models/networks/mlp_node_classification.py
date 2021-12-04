import torch.nn as nn
import torch.nn.functional as F

from .modules import Residual
from .utils.dropout_layers import dropout_layers


def create_network(num_features, num_classes, opt):
    return MLPNodeClassification(
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


class MLPNodeClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, ffn_dim, n_layers, prob_survival):
        super().__init__()
        self.prob_survival = prob_survival

        assert n_layers >= 2
        self.emgbedding = nn.Linear(num_features, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.layers += [Residual(MLPBlock(hidden_dim, ffn_dim))]
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.emgbedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x = data.x

        x = self.emgbedding(x)
        prob_survival = self.prob_survival if self.training else 1
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x)
        x = self.classifier(x)

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
