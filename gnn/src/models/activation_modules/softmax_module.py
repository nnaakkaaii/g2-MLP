import argparse

import torch
import torch.nn as nn


def create_activation_module(opt: argparse.Namespace) -> nn.Module:
    return Softmax(opt.out_dim, opt.last_p_dim)


class Softmax(nn.Module):
    """fc層を1層結合 (softmaxはlossに含まれる)
    """
    def __init__(self, out_dim: int, last_p_dim: int = 100) -> None:
        super().__init__()
        model = [
            nn.Linear(last_p_dim, out_dim),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
