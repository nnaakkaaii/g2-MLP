import torch.nn as nn


def create_norm_module(input_size: int) -> nn.Module:
    return nn.BatchNorm2d(input_size, affine=True, track_running_stats=True)
