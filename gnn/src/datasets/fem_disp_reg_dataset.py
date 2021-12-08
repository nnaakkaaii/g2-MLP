import argparse
import os
from typing import Any

from src.transforms.disp_as_label_transform import create_transform
from torch_geometric.data import InMemoryDataset

from .fem import FemDataset


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = FemDataset(opt.data_dir, split='train' if is_train else 'test', transform=transform, pre_transform=create_transform(opt))
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'FEM_DISP_REG'), help='FEMデータを保存する場所')
    return parser
