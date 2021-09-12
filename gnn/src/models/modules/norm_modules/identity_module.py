import torch
import torch.nn as nn


def create_norm_module(input_size: int) -> nn.Module:
    return IdentityModule()


class IdentityModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
