import argparse
import os
from typing import Any

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import Planetoid


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = Planetoid(opt.data_dir, name='CiteSeer', split='public', transform=transform)
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'CiteSeer'), help='CiteSeerデータを保存する場所')
    return parser
