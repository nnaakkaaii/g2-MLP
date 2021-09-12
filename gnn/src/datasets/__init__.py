import argparse
from typing import Any, Callable, Dict

from . import mnist_dataset
from .base_dataset import BaseDataset

datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], BaseDataset]] = {
    'mnist': mnist_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'mnist': mnist_dataset.dataset_modify_commandline_options,
}
