import torch.nn as nn

from .residual import Residual
from .gmlp_layer import gMLPLayer
from ..utils.dropout_layers import dropout_layers


class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_layers, prob_survival=1.):
        super().__init__()
        assert n_layers >= 1

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers += [Residual(gMLPLayer(hidden_dim, ffn_dim))]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        prob_survival = self.prob_survival if self.training else 1
        
        xs = []
        for layer in dropout_layers(self.layers, prob_survival):
            x = layer(x, edge_index)
            xs += [x]

        return x, xs
