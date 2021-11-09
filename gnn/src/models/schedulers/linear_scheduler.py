import argparse
from typing import Any

from torch.optim import lr_scheduler


def modify_scheduler_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--n_epochs_decay', type=int, default=500, help='lrを減少させ始めるepoch数')
    return parser


def create_scheduler(optimizer: Any, opt: argparse.Namespace) -> Any:
    def __lr_lambda(epoch: int) -> float:
        return 1.0 - max(0, epoch - opt.n_epochs_decay) / float(opt.n_epochs - opt.n_epochs_decay)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=__lr_lambda)
