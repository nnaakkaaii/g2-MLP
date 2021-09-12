import argparse

import torch
from torch.nn import init


def init_weight(data: torch.Tensor, opt: argparse.Namespace) -> None:
    init.kaiming_normal_(data, a=0, mode='fan_in')
    return
