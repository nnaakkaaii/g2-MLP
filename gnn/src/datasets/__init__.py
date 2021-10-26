import argparse
from typing import Any, Callable, Dict

import torch.utils.data as data

from . import tu_dataset

datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], data.Dataset]] = {
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
