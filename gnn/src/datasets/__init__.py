import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import ppi_dataset, cora_dataset

datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], InMemoryDataset]] = {
    'ppi': ppi_dataset.create_dataset,
    'cora': cora_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'ppi': ppi_dataset.dataset_modify_commandline_options,
    'cora': cora_dataset.dataset_modify_commandline_options,
}
