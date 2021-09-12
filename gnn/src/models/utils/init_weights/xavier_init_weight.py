import argparse

import torch
from torch.nn import init


def init_weight_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling Factor')
    return parser


def init_weight(data: torch.Tensor, opt: argparse.Namespace) -> None:
    init.xavier_normal_(data, gain=opt.init_gain)
    return
