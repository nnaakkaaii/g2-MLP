import argparse
import os
from typing import Any, Dict

import torch
from torchvision.datasets import MNIST

from . import base_dataset


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return MnistDataset(transform, opt.max_dataset_size, opt.img_dir, is_train, opt.train_ratio)


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--img_dir', type=str, default=os.path.join('inputs', 'mnist'), help='mnistデータを保存する場所')
    return parser


class MnistDataset(base_dataset.BaseDataset):
    """torchvisionのMNISTを利用したデータセット
    MNISTのDatasetの並び順が変更しないというメタ知識を前提としている
    渡すtransformも変更する
    """
    def __init__(self, transform: Any, max_dataset_size: int, img_dir: str, is_train: bool, train_ratio: float) -> None:
        dataset = MNIST(img_dir, download=True, transform=transform)
        if is_train:
            self.dataset = dataset[:int(len(dataset) * train_ratio)]
        else:
            self.dataset = dataset[int(len(dataset) * train_ratio):]

        super().__init__(max_dataset_size, len(self.dataset), is_train)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, t = self.dataset[idx]
        return {'x': x, 't': t}
