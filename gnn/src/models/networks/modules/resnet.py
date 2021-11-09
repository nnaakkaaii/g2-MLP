import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)

    def reset_parameters(self):
        if hasattr(self.fn, 'reset_parameters'):
            self.fn.reset_parameters()
