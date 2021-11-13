import argparse
from typing import Any

import torch
from torch_geometric.data import Data


def create_transform(opt: argparse.Namespace) -> Any:
    return ClsToken()


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class ClsToken:
    """Add [CLS] token.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index

        num_nodes, num_features = x.shape

        cls_node_x = torch.zeros((1, num_features))
        nodes_to_cls_node = torch.stack((torch.arange(num_nodes), torch.zeros(num_nodes))).to(torch.int64)

        new_data = data.clone()

        new_data.x = torch.cat((cls_node_x, x))
        new_data.edge_index = torch.cat((nodes_to_cls_node, edge_index), dim=1)

        return new_data

    def __repr__(self):
        return f'{self.__class__.__name__}'
