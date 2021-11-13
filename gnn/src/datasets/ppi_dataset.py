import argparse
import os
from typing import Any

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import PPI


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = PPI(opt.data_dir, split='train' if is_train else 'test', transform=transform)
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'PPI'), help='PPIデータを保存する場所')
    return parser
