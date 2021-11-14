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
        self.dropout_rate = dropout_rate
        
        self.embedding = nn.Linear(num_features, hidden_dim)

        self.conv1 = gMLPBlock(hidden_dim, ffn_dim, n_layers, prob_survival)
        self.conv2 = gMLPBlock(hidden_dim, ffn_dim, n_layers, prob_survival)
        self.conv3 = gMLPBlock(hidden_dim, ffn_dim, n_layers, prob_survival)
        
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x)

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return self.linear3(x)



class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_layers, prob_survival):
        super().__init__()
        self.prob_survival = prob_survival

        assert n_layers >= 2
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPLayer(hidden_dim, ffn_dim))]

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        prob_survival = self.prob_survival if self.training else 1

        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x, edge_index)
        
        return x
