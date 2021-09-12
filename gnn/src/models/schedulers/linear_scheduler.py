import argparse
from typing import Any

from torch.optim import lr_scheduler


def create_scheduler(optimizer: Any, opt: argparse.Namespace) -> Any:
    def __lr_lambda(epoch: int) -> float:
        return 1.0 - max(0, epoch + opt.epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=__lr_lambda)
