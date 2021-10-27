import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import bce_loss, mce_loss, mse_loss, nll_loss

losses: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'mce': mce_loss.create_loss,
    'bce': bce_loss.create_loss,
    'nll': nll_loss.create_loss,
    'mse': mse_loss.create_loss,
}

loss_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'mce': lambda x: x,
    'bce': lambda x: x,
    'nll': lambda x: x,
    'mse': lambda x: x,
}
