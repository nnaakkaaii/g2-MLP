import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import dd_dataset, mutag_dataset, ppi_dataset, proteins_dataset

datasets: Dict[str, Callable[[Any, bool, int, argparse.Namespace], InMemoryDataset]] = {
    'ppi': ppi_dataset.create_dataset,
    'dd': dd_dataset.create_dataset,
    'mutag': mutag_dataset.create_dataset,
    'proteins': proteins_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'ppi': ppi_dataset.dataset_modify_commandline_options,
    'dd': dd_dataset.dataset_modify_commandline_options,
    'mutag': mutag_dataset.dataset_modify_commandline_options,
    'proteins': proteins_dataset.dataset_modify_commandline_options,
}
