import argparse
import os
from typing import Any

import numpy as np
import torch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import TUDataset


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = TUDataset(opt.data_dir, name='PROTEINS', transform=transform)
    index_file_name = '{}-idx.txt'.format('train' if is_train else 'test')
    indices = torch.as_tensor(np.loadtxt(os.path.join(opt.data_dir, index_file_name), dtype=np.int32), dtype=torch.long)
    dataset = dataset[indices]
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'PROTEINS'), help='PROTEINSデータを保存する場所')
    return parser
