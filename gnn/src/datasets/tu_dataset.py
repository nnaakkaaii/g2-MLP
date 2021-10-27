import argparse
import os
from typing import Any

import numpy as np
import torch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import TUDataset


def create_dataset(transform: Any, is_train: bool, fold_number: int, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = TUDataset(opt.data_dir, name=opt.dataset_name, transform=transform, use_node_attr=True)
    index_file_name = '{}_idx-{:02}.txt'.format('train' if is_train else 'test', fold_number)
    indices = torch.as_tensor(np.loadtxt(os.path.join(opt.index_file_dir, index_file_name), dtype=np.int32), dtype=torch.long)
    return dataset[indices]


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'DD'), help='TUデータを保存する場所')
    parser.add_argument('--index_file_dir', type=str, default=os.path.join('inputs', 'DD', '10fold_idx'), help='TUデータのindexファイルを保存するディレクトリ')
    return parser
