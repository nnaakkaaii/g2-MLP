import argparse
from typing import List

import torch
import torch.nn as nn


def create_module(opt: argparse.Namespace) -> nn.Module:
    return FCModule(opt.in_size, opt.in_nch, opt.last_p_dim, list(map(int, opt.n_layers.split(','))))


def module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--n_layers', type=str, default='1024,516', help='各層のパーセプトロン数')
    return parser


class FCModule(nn.Module):
    """Full Connected Module
    入力 : opt.in_size, opt.in_size, opt.in_nch
    出力 : opt.out_dim
    """
    def __init__(self,
                 in_size: int,
                 in_nch: int,
                 out_dim: int,
                 n_layers: List[int]) -> None:
        """
        :param in_size:
        :param in_nch:
        :param out_dim:
        :param n_layers:
        """
        super().__init__()

        model = []
        for i, n_layer in enumerate(n_layers):
            if i == 0:
                model += self.__block(in_size * in_size * in_nch, n_layer)
            else:
                model += self.__block(n_layers[i - 1], n_layer)
        model += [nn.Linear(n_layers[-1], out_dim)]

        self.model = nn.Sequential(*model)

    @staticmethod
    def __block(in_size: int, out_size: int) -> List[nn.Module]:
        return [
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.model(x)
