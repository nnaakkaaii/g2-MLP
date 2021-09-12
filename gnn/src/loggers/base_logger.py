import abc
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List

from ..models.base_model import AbstractModel
from .abstract_logger import AbstractLogger
from .utils.averager import Averager


def logger_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class BaseLogger(AbstractLogger, metaclass=abc.ABCMeta):
    """標準的なDataLoaderの実装
    trainとvalで別のaveragerに計上していく
    print_statusでtrainとvalを表示を分岐する
    """
    def __init__(self, model: AbstractModel, opt: argparse.Namespace) -> None:
        super().__init__(model=model, opt=opt)
        self.save_dir = model.save_dir
        self._trained_epoch = opt.epoch
        self._train_dataset_length = -1
        self._val_dataset_length = -1
        self.train_averager = Averager()
        self.val_averager = Averager()

        self.history: DefaultDict[str, List[float]] = defaultdict(list)
        os.makedirs(self.save_dir, exist_ok=True)

    def start_epoch(self) -> None:
        self.train_averager.reset()
        self.val_averager.reset()
        return

    def end_epoch(self) -> None:
        self._increment_epoch()
        return

    def end_train_iter(self) -> None:
        current_losses = self.model.get_current_loss_and_metrics()
        self.train_averager.send(current_losses)
        return

    def end_val_iter(self) -> None:
        current_losses = self.model.get_current_loss_and_metrics()
        self.val_averager.send(current_losses)
        return

    def save_options(self) -> None:
        with open(os.path.join(self.save_dir, 'options.json'), 'w') as f:
            json.dump(self.opt, f)
        return

    def set_dataset_length(self, train_dataset_length: int, val_dataset_length) -> None:
        self._train_dataset_length = train_dataset_length
        self._val_dataset_length = train_dataset_length
        return

    def _increment_epoch(self) -> None:
        self._trained_epoch += 1
        return

    def print_status(self, iterations: int, status_dict: Dict[str, float], is_train: bool) -> None:
        if is_train:
            status_str = f'[Epoch {self._trained_epoch}][{iterations:.0f}/{self._train_dataset_length}] '
            for key, value in status_dict.items():
                status_str += f'train_{key}: {value:.4f}, '
        else:
            status_str = f'[Epoch {self._trained_epoch}][{iterations:.0f}/{self._val_dataset_length}] '
            for key, value in status_dict.items():
                status_str += f'val_{key}: {value:.4f}, '
        sys.stdout.write(f'\r{status_str}')
        sys.stdout.flush()
        return
