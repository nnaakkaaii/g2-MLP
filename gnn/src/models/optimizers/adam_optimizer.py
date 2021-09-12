import argparse
from typing import Any

import torch.optim as optim


def optimizer_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    return parser


def create_optimizer(params: Any, opt: argparse.Namespace) -> Any:
    return optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
