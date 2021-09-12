from typing import Callable, Dict

import torch.nn as nn

from . import (batch_norm_1d_module, batch_norm_2d_module, identity_module,
               instance_norm_module)

norm_modules: Dict[str, Callable[[int], nn.Module]] = {
    'none': identity_module.create_norm_module,
    'batch_norm_1d': batch_norm_1d_module.create_norm_module,
    'batch_norm_2d': batch_norm_2d_module.create_norm_module,
    'instance_norm': instance_norm_module.create_norm_module,
}
