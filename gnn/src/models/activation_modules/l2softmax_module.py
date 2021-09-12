import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f


def activation_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--last_temperature', type=float, default=16, help='softmaxの温度')
    return parser


def create_activation_module(opt: argparse.Namespace) -> nn.Module:
    return L2Softmax(opt.out_dim, opt.last_p_dim, opt.last_temperature)


class L2Softmax(nn.Module):
    """fc層を1層 + L2Norm層を結合 (softmaxはlossに含まれる)
    """
    def __init__(self, out_dim: int, last_p_dim: int = 100, last_alpha: float = 16) -> None:
        super().__init__()
        model = [
            nn.Linear(last_p_dim, out_dim),
            L2Norm(last_alpha)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class L2Norm(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.__alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = f.normalize(x, p=2, dim=1).float()
        return y * self.__alpha
