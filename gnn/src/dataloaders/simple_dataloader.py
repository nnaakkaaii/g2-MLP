import argparse
from typing import Any

import torch.utils.data as data

from ..datasets.base_dataset import BaseDataset


def create_dataloader(dataset: BaseDataset, opt: argparse.Namespace) -> data.DataLoader:
    return SimpleDataLoader(
        dataset=dataset, batch_size=opt.batch_size, serial_batches=opt.serial_batches, num_threads=opt.num_threads)


def dataloader_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--serial_batches', action='store_true', help='dataloaderの読み込み順をランダムにするか')
    parser.add_argument('--num_threads', type=int, default=0, help='データローダーの並列数')
    return parser


class SimpleDataLoader(data.DataLoader):
    """DataLoaderの標準的な実装
    trainでない時はshuffleしない
    """
    def __init__(self,
                 dataset: BaseDataset,
                 batch_size: int,
                 serial_batches: bool,
                 num_threads: int) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=not serial_batches if dataset.is_train else False,
            num_workers=int(num_threads),
        )

        self._batch_size = batch_size
        self.max_dataset_size = dataset.max_dataset_size

    def __iter__(self) -> Any:
        for i, iter_data in enumerate(self):
            if i * self._batch_size >= self.max_dataset_size:
                break
            yield iter_data
