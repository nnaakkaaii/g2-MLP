import argparse
import os
from typing import Any

import numpy as np
import torch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import PPI


def create_dataset(transform: Any, is_train: bool, fold_number: int, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = PPIDataset(opt.data_dir, transform=transform)
    index_file_name = '{}_idx-{:02}.txt'.format('train' if is_train else 'test', fold_number)
    indices = torch.as_tensor(np.loadtxt(os.path.join(opt.index_file_dir, index_file_name), dtype=np.int32), dtype=torch.long)
    return dataset[indices]


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'PPI'), help='PPIデータを保存する場所')
    parser.add_argument('--index_file_dir', type=str, default=os.path.join('inputs', 'PPI', '10fold_idx'), help='PPIデータのindexファイルを保存するディレクトリ')
    return parser


class PPIDataset(PPI):
    def __init__(self, root: str, transform: Any = None, pre_transform: Any = None, pre_filter: Any = None) -> None:
        super().__init__(root, split='train', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        train_dataset = PPI(root, split='train', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        val_dataset = PPI(root, split='val', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        test_dataset = PPI(root, split='test', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        self.data, self.slices = InMemoryDataset.collate(list(train_dataset) + list(val_dataset) + list(test_dataset))
