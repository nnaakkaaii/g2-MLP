import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import tu_dataset

datasets: Dict[str, Callable[[Any, bool, int, argparse.Namespace], InMemoryDataset]] = {
    'DD': tu_dataset.create_dataset,
    'PTC_MR': tu_dataset.create_dataset,
    'NCI1': tu_dataset.create_dataset,
    'PROTEINS': tu_dataset.create_dataset,
    'IMDB-BINARY': tu_dataset.create_dataset,
    'IMDB-MULTI': tu_dataset.create_dataset,
    'MUTAG': tu_dataset.create_dataset,
    'COLLAB': tu_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'DD': tu_dataset.dataset_modify_commandline_options,
    'PTC_MR': tu_dataset.dataset_modify_commandline_options,
    'NCI1': tu_dataset.dataset_modify_commandline_options,
    'PROTEINS': tu_dataset.dataset_modify_commandline_options,
    'IMDB-BINARY': tu_dataset.dataset_modify_commandline_options,
    'IMDB-MULTI': tu_dataset.dataset_modify_commandline_options,
    'MUTAG': tu_dataset.dataset_modify_commandline_options,
    'COLLAB': tu_dataset.dataset_modify_commandline_options,
}
