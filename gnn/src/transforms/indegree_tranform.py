from argparse import ArgumentParser, Namespace
from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def create_transform(opt: Namespace) -> Any:
    return Indegree(not opt.transform_no_norm, opt.max_value)


def transform_modify_commandline_option(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--transform_no_norm', action='store_true', help='transformで加える次数の特徴に対してnormalizeを行わない場合は指定する')
    parser.add_argument('--max_value', type=float, default=None, help='normalizeを行うときの分母を予め指定したい場合')
    return parser


class Indegree:
    """
    normalizedされたnodeの次数をnode featureの1つとして加える

    参考実装 : https://github.com/leftthomas/DGCNN/blob/master/utils.py
    """

    def __init__(self, norm: bool = True, max_value: Optional[float] = None) -> None:
        self.norm = norm
        self.max_value = max_value

    def __call_(self, data: Data) -> Data:
        col, x = data.edge_index[1], data.x
        deg = degree(col, col, data.num_nodes)

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
