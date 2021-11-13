import argparse
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def create_transform(opt: argparse.Namespace) -> Any:
    return Indegree()


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class Indegree:
    """
    normalizedされたnodeの次数をnode featureの1つとして加える

    参考実装 : https://github.com/leftthomas/DGCNN/blob/master/utils.py
    """

    def __init__(self) -> None:
        self.norm = True
        self.max_value = None

    def __call__(self, data: Data) -> Data:
        col, x = data.edge_index[1], data.x
        deg = degree(col, data.num_nodes)

        if self.norm:
            deg = deg / (deg.max() if self.max_value is None else self.max_value)

        deg = deg.view(-1, 1)

        new_data = data.clone()

        if x is not None:
            x = x.view(-1, 1) if x.dim() == 1 else x
            new_data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            new_data.x = deg

        return new_data

    def __repr__(self):
        return f'{self.__class__.__name__}(norm={self.norm}, max_value={self.max_value})'
