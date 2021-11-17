import argparse
from typing import Any, Callable, Dict

from torch_geometric.data.in_memory_dataset import InMemoryDataset

from . import (citeseer_dataset, cora_dataset, dd_dataset, enzymes_dataset,
               frankenstein_dataset, mutag_dataset, nci1_dataset,
               nci109_dataset, ppi_dataset, proteins_dataset, pubmed_dataset,
               imdb_binary_dataset, imdb_multi_dataset, collab_dataset, ptc_mr_dataset)

datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], InMemoryDataset]] = {
    'ppi': ppi_dataset.create_dataset,
    'cora': cora_dataset.create_dataset,
    'citeseer': citeseer_dataset.create_dataset,
    'pubmed': pubmed_dataset.create_dataset,
    'dd': dd_dataset.create_dataset,
    'enzymes': enzymes_dataset.create_dataset,
    'frankenstein': frankenstein_dataset.create_dataset,
    'mutag': mutag_dataset.create_dataset,
    'nci1': nci1_dataset.create_dataset,
    'nci109': nci109_dataset.create_dataset,
    'proteins': proteins_dataset.create_dataset,
    'imdb_binary': imdb_binary_dataset.create_dataset,
    'imdb_multi': imdb_multi_dataset.create_dataset,
    'collab': collab_dataset.create_dataset,
    'ptc_mr': ptc_mr_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'ppi': ppi_dataset.dataset_modify_commandline_options,
    'cora': cora_dataset.dataset_modify_commandline_options,
    'citeseer': citeseer_dataset.dataset_modify_commandline_options,
    'pubmed': pubmed_dataset.dataset_modify_commandline_options,
    'dd': dd_dataset.dataset_modify_commandline_options,
    'enzymes': enzymes_dataset.dataset_modify_commandline_options,
    'frankenstein': frankenstein_dataset.dataset_modify_commandline_options,
    'mutag': mutag_dataset.dataset_modify_commandline_options,
    'nci1': nci1_dataset.dataset_modify_commandline_options,
    'nci109': nci109_dataset.dataset_modify_commandline_options,
    'proteins': proteins_dataset.dataset_modify_commandline_options,
    'imdb_binary': imdb_binary_dataset.dataset_modify_commandline_options,
    'imdb_multi': imdb_multi_dataset.dataset_modify_commandline_options,
    'collab': collab_dataset.dataset_modify_commandline_options,
    'ptc_mr': ptc_mr_dataset.dataset_modify_commandline_options,
}
