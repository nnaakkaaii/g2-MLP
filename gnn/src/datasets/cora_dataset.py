import os
import argparse
from typing import Any

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import Planetoid


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = Planetoid(opt.data_dir, name='Cora', split='public', transform=transform)
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'Cora'), help='Coraデータを保存する場所')
    return parser
