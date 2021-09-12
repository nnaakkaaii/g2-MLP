import abc
import argparse
import os
from typing import Any, Dict, Union

import torch.nn as nn


class AbstractModel(metaclass=abc.ABCMeta):
    """This class is an abstract class for models.
    """
    modules: Dict[str, Union[nn.Module, nn.DataParallel]] = {}
    criterion: nn.Module
    optimizer: Any
    scheduler: Any

    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialize the BaseModel class.
        """
        self.opt = opt
        self.save_dir = os.path.join(opt.save_dir, opt.name)

    @abc.abstractmethod
    def setup(self, opt: argparse.Namespace) -> None:
        """Called in construct
        Setup (Load and print networks)
            -- load networks    : if not training mode or continue_train is True, then load opt.epoch
            -- print networks
        """
        pass

    @abc.abstractmethod
    def set_input(self, input_: Dict[str, Any]) -> None:
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        pass

    @abc.abstractmethod
    def forward(self) -> None:
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        pass

    @abc.abstractmethod
    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights; called in every training iteration.
        """
        pass

    @abc.abstractmethod
    def test(self) -> None:
        """ Forward function used in test time.
        """
        pass

    @abc.abstractmethod
    def train_mode(self) -> None:
        """turn to train mode
        """
        pass

    @abc.abstractmethod
    def eval_mode(self) -> None:
        """make models eval mode during test time.
        """
        pass

    @abc.abstractmethod
    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks; called at the end of every epoch
        """
        pass

    @abc.abstractmethod
    def get_current_loss_and_metrics(self) -> Dict[str, float]:
        """Return training losses / errors. train_option.py will print out these errors on console, and save them to a file
        """
        pass

    @abc.abstractmethod
    def save_networks(self, epoch: Union[int, str]) -> None:
        """Save all the networks to the disk.
        """
        pass

    @abc.abstractmethod
    def load_networks(self, epoch: int) -> None:
        """Load all the networks from the disk.
        """
        pass

    @abc.abstractmethod
    def print_networks(self, verbose: bool) -> None:
        """Print the total number of parameters in the network and (if verbose) network architecture
        """
        pass
