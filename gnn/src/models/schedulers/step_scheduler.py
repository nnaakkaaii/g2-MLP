import argparse
from typing import Any

from torch.optim import lr_scheduler


def modify_scheduler_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--lr_decay_iters', type=int, default=200, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='multiply by a gamma every lr_decay_iters iterations')
    return parser


def create_scheduler(optimizer: Any, opt: argparse.Namespace) -> Any:
    return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
