import argparse
from typing import Any, Tuple

import torch
from torch_geometric.data import Data


def create_transform(opt: argparse.Namespace) -> Any:
    return LabelNormalize(mean=tuple(opt.mean), std=tuple(opt.std))


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    opt, _ = parser.parse_known_args()
    if not hasattr(opt, 'mean'):
        parser.add_argument('--mean', nargs='+', type=float, required=True, help='データの平均値')
    if not hasattr(opt, 'std'):
        parser.add_argument('--std', nargs='+', type=float, required=True, help='データの標準誤差')
    return parser


class LabelNormalize:

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        for i in range(3):
            data.y[:, i] = (data.y[:, i] - self.mean[i]) / self.std[i]
        return data

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        for i in range(3):
            y[:, i] = y[:, i] * self.std[i] + self.mean[i]
        return y
