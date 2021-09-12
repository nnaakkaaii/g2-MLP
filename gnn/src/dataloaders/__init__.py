import argparse
from typing import Callable, Dict

import torch.utils.data as data

from ..datasets.base_dataset import BaseDataset
from . import simple_dataloader

dataloaders: Dict[str, Callable[[BaseDataset, argparse.Namespace], data.DataLoader]] = {
    'simple': simple_dataloader.create_dataloader,
}

dataloader_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_dataloader.dataloader_modify_commandline_options,
}
