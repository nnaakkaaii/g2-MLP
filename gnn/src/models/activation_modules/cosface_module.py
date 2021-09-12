import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f

from . import _xface_module


def activation_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--last_temperature', type=float, default=3, help='softmaxの温度')
    parser.add_argument('--last_margin', type=float, default=0.4, help='cosfaceのマージンの幅')
    return parser


def create_activation_module(opt: argparse.Namespace) -> nn.Module:
    return _xface_module.XFace(AddMarginProduct(opt.last_temperature, opt.last_margin), opt.out_dim, opt.last_p_dim)


class AddMarginProduct(_xface_module.BaseMarginProduct):
    def __init__(self, temperature: float, margin: float) -> None:
        super().__init__()
        self.__temperature = temperature
        self.__margin = margin

    def forward(self, x: torch.Tensor, t: torch.Tensor, is_train: bool) -> torch.Tensor:
        assert self.weight.size(0) > 0
        cosine = f.linear(f.normalize(x), f.normalize(self.weight)).float()
        if self.training:
            phi = cosine - self.__margin
            output = (t * phi + (1.0 - t) * cosine) * self.__temperature
            return output
        else:
            return cosine * self.__temperature
