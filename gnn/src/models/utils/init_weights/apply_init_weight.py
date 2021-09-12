import argparse
from typing import Callable

import torch
import torch.nn as nn

from . import normal_init_weight


def apply_init_weight(net: nn.Module, opt: argparse.Namespace, init_weight: Callable[[torch.Tensor, argparse.Namespace], None]) -> None:
    """Initialize network weights
    :param net:
    :param opt:
    :param init_weight:
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init_weight(m.weight.data, opt)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            normal_init_weight.init_weight(m.weight.data, opt)
            nn.init.constant_(m.bias.data, 0.0)
        return

    net.apply(init_func)
    return
