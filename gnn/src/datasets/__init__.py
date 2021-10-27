import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import ppi_dataset

datasets: Dict[str, Callable[[Any, bool, int, argparse.Namespace], InMemoryDataset]] = {
    'PPI': ppi_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'PPI': ppi_dataset.dataset_modify_commandline_options,
}
