import argparse
from typing import Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.loader import DataLoader

from . import simple_dataloader

dataloaders: Dict[str, Callable[[InMemoryDataset, bool, argparse.Namespace], DataLoader]] = {
    'simple': simple_dataloader.create_dataloader,
}

dataloader_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_dataloader.dataloader_modify_commandline_options,
}
