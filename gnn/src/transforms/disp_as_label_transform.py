import argparse
from typing import Any

from torch_geometric.data import Data


def create_transform(opt: argparse.Namespace) -> Any:
    return DispAsLabel()


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class DispAsLabel:

    def __init__(self) -> None:
        pass

    def __call__(self, data: Data) -> Data:
        data.y = data.x
        return data
