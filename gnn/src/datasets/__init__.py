import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import ppi_dataset, dd_dataset, mutag_dataset, proteins_dataset

datasets: Dict[str, Callable[[Any, bool, int, argparse.Namespace], InMemoryDataset]] = {
    'PPI': ppi_dataset.create_dataset,
    'DD': dd_dataset.create_dataset,
    'MUTAG': mutag_dataset.create_dataset,
    'PROTEINS': proteins_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'PPI': ppi_dataset.dataset_modify_commandline_options,
    'DD': dd_dataset.dataset_modify_commandline_options,
    'MUTAG': mutag_dataset.dataset_modify_commandline_options,
    'PROTEINS': proteins_dataset.dataset_modify_commandline_options,
}
