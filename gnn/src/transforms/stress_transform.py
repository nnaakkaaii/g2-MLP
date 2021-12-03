import argparse
from typing import Any

import torch
from torch_geometric.data import Data


def create_transform(opt: argparse.Namespace) -> Any:
    return Stress()


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class Stress:
    def __init__(self) -> None:
        self.xy = True
        self.base = 2
        self.exponent = 5

    def __call__(self, data: Data) -> Data:
        if self.xy:
            y = torch.sqrt(data.y[:, 0]**2 + data.y[:, 1]**2)
        else:
            y = torch.sqrt(data.y[:, 0]**2 + data.y[:, 1]**2 + data.y[:, 2]**2)
        y = torch.bucketize(y, torch.tensor([self.base ** i for i in range(self.exponent)]))
        data.y = y
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(xy={self.xy}, base={self.base}, exponent={self.exponent})'
