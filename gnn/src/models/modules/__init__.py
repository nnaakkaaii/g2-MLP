import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import fc_module

modules: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'fc': fc_module.create_module,
}

module_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'fc': fc_module.module_modify_commandline_options,
}
