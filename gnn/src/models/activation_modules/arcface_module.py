import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as f

from . import _xface_module


def activation_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--last_temperature', type=float, default=3, help='softmaxの温度')
    parser.add_argument('--last_margin', type=float, default=0.5, help='arcfaceのマージンの角度')
    return parser


def create_activation_module(opt: argparse.Namespace) -> nn.Module:
    return _xface_module.XFace(ArcMarginProduct(opt.last_temperature, opt.last_margin), opt.out_dim, opt.last_p_dim)


class ArcMarginProduct(_xface_module.BaseMarginProduct):
    def __init__(self, temperature: float, theta: float) -> None:
        super().__init__()
        self.__temperature = temperature
        self.__cos_m = math.cos(theta)
        self.__sin_m = math.sin(theta)

    def forward(self, x: torch.Tensor, t: torch.Tensor, is_train: bool) -> torch.Tensor:
        assert self.weight.size(0) > 0
        cosine = f.linear(f.normalize(x), f.normalize(self.weight)).float()

        if is_train:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.__cos_m - sine * self.__sin_m
            phi = torch.where(cosine > 0, phi, cosine)
            output = (t * phi + (1.0 - t) * cosine) * self.__temperature
            return output
        else:
            return cosine * self.__temperature
