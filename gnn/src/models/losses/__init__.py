import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import cross_entropy_loss, mse_loss, nll_loss

losses: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'cross_entropy': cross_entropy_loss.create_loss,
    'nll': nll_loss.create_loss,
    'mse': mse_loss.create_loss,
}

loss_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'cross_entropy': lambda x: x,
    'nll': lambda x: x,
    'mse': lambda x: x,
}
