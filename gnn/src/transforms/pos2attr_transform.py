import argparse
from typing import Any

import torch
from torch_geometric.data import Data


def create_transform(opt: argparse.Namespace) -> Any:
    return Pos2Attr()


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class Pos2Attr:

    def __init__(self) -> None:
        pass

    def __call__(self, data: Data) -> Data:
        data.x = torch.cat([data.x, data.pos], dim=1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(xy={self.xy}, base={self.base}, exponent={self.exponent})'
