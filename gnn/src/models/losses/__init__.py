import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import bce_loss, mce_loss, mse_loss, nll_loss

losses: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'bce': bce_loss.create_loss,
    'mce': mce_loss.create_loss,
    'mse': mse_loss.create_loss,
    'nll': nll_loss.create_loss,
}

loss_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'bce': lambda x: x,
    'mce': lambda x: x,
    'mse': lambda x: x,
    'nll': lambda x: x,
}
