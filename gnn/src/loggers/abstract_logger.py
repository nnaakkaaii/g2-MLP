import abc
import argparse
from typing import Any, Callable, Dict, Optional

import torch.nn as nn


class AbstractLogger(metaclass=abc.ABCMeta):
    def __init__(self, opt: argparse.Namespace) -> None:
        self.opt = opt

        self._test_callback: Optional[Callable[[], None]] = None
        self._network: Optional[nn.Module] = None

    def set_test_callback(self, test_callback: Callable[[], None]) -> None:
        pass

    def set_network(self, network: nn.Module) -> None:
        pass

    @abc.abstractmethod
    def on_start(self, state: Dict[str, Any]) -> None:
        return

    @abc.abstractmethod
    def on_end(self, state: Dict[str, Any]) -> None:
        return

    @abc.abstractmethod
    def on_sample(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def on_forward(self, state: Dict[str, Any]) -> None:
        return

    @abc.abstractmethod
    def on_start_epoch(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def on_end_epoch(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def on_end_all_training(self) -> None:
        pass

    @abc.abstractmethod
    def save_models(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def save_options(self) -> None:
        pass

    @abc.abstractmethod
    def print_status(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def print_networks(self) -> None:
        pass

    @property
    def hooks(self) -> Dict[str, Callable[[Dict[str, Any]], None]]:
        return {
            'on_start': self.on_start,
            'on_end': self.on_end,
            'on_sample': self.on_sample,
            'on_forward': self.on_forward,
            'on_start_epoch': self.on_start_epoch,
            'on_end_epoch': self.on_end_epoch,
        }
