import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import (gat_product_3d_attention_module,
               scaled_dot_product_3d_attention_module)

# attention_hidden_dim, opt
attention_modules: Dict[str, Callable[[int, argparse.Namespace], nn.Module]] = {
    'scaled_dot_3d': scaled_dot_product_3d_attention_module.create_attention_module,
    'gat_product_3d': gat_product_3d_attention_module.create_attention_module,
}

attention_module_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'scaled_dot_3d': scaled_dot_product_3d_attention_module.attention_module_modify_commandline_options,
    'gat_product_3d': gat_product_3d_attention_module.attention_module_modify_commandline_options,
}
