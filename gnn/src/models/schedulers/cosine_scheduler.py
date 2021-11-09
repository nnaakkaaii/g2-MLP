import argparse
from typing import Any

from torch.optim import lr_scheduler


def create_scheduler(optimizer: Any, opt: argparse.Namespace) -> Any:
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
