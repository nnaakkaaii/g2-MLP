import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import softmax_module, l2softmax_module, arcface_module, cosface_module


activation_modules: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'softmax': softmax_module.create_activation_module,
    'l2softmax': l2softmax_module.create_activation_module,
    'arcface': arcface_module.create_activation_module,
    'cosface': cosface_module.create_activation_module,
}

activation_module_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'softmax': lambda x: x,
    'l2softmax': l2softmax_module.activation_module_modify_commandline_options,
    'arcface': arcface_module.activation_module_modify_commandline_options,
    'cosface': cosface_module.activation_module_modify_commandline_options,
}
