import torch.nn as nn


def create_norm_module(input_size: int) -> nn.Module:
    return nn.InstanceNorm2d(input_size, affine=False, track_running_stats=False)
