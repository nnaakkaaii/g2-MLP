import argparse
import os
from typing import Any

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import PPI


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = PPIDataset(opt.data_dir, split='train' if is_train else 'test', transform=transform)
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'PPI'), help='PPIデータを保存する場所')
    return parser


class PPIDataset(PPI):
    def __init__(self, root: str, split: str, transform: Any = None, pre_transform: Any = None, pre_filter: Any = None) -> None:
        if split == 'train':
            # 学習時はtrainとvalidationを合わせて学習を行う
            super().__init__(root, split='train', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
            train_dataset = PPI(root, split='train', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
            val_dataset = PPI(root, split='val', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
            self.data, self.slices = InMemoryDataset.collate(list(train_dataset) + list(val_dataset))
        elif split == 'val':
            # チューニング時はvalidationを用いる
            super().__init__(root, split='val', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        elif split == 'test':
            super().__init__(root, split='test', transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        else:
            raise KeyError('split should be in [train/val/test]')
