import abc
import argparse
from typing import Dict

from ..models.base_model import AbstractModel


class AbstractLogger(metaclass=abc.ABCMeta):
    def __init__(self, model: AbstractModel, opt: argparse.Namespace) -> None:
        self.model = model
        self.opt = vars(opt)

    @abc.abstractmethod
    def start_epoch(self) -> None:
        pass

    @abc.abstractmethod
    def end_epoch(self) -> None:
        pass

    @abc.abstractmethod
    def end_train_iter(self) -> None:
        pass

    @abc.abstractmethod
    def end_val_iter(self) -> None:
        pass

    @abc.abstractmethod
    def end_all_training(self) -> None:
        pass

    @abc.abstractmethod
    def save_options(self) -> None:
        pass

    @abc.abstractmethod
    def set_dataset_length(self, train_dataset_length: int, val_dataset_length: int) -> None:
        pass

    @abc.abstractmethod
    def print_status(self, iterations: int, status_dict: Dict[str, float], is_train: bool) -> None:
        pass
