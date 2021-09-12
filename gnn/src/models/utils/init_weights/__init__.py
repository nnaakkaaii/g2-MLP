import argparse
from typing import Callable, Dict

import torch

from . import (kaiming_init_weight, normal_init_weight, orthogonal_init_weight,
               xavier_init_weight)

init_weights: Dict[str, Callable[[torch.Tensor, argparse.Namespace], None]] = {
    'normal': normal_init_weight.init_weight,
    'kaiming': kaiming_init_weight.init_weight,
    'xavier': xavier_init_weight.init_weight,
    'orthogonal': orthogonal_init_weight.init_weight,
}

init_weight_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'normal': normal_init_weight.init_weight_modify_commandline_options,
    'kaiming': lambda x: x,
    'xavier': xavier_init_weight.init_weight_modify_commandline_options,
    'orthogonal': orthogonal_init_weight.init_weight_modify_commandline_options,
}
