import torch.nn as nn

from .modules.gmlp_block import gMLPBlock


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

        assert n_layers >= 2
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.conv = gMLPBlock(hidden_dim, ffn_dim, n_layers, prob_survival=prob_survival)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x)
        x = self.conv(x, edge_index)
        x, _ = self.classifier(x)

        return x
