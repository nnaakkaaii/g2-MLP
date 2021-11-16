import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import (
    gat_node_classification,
    gcn_node_classification,
    gmlp_node_classification,
    mlp_node_classification,
    gmlp_graph_classification,
    gmlp_sagpool_graph_classification,
)

networks: Dict[str, Callable[[int, int, bool, argparse.Namespace], nn.Module]] = {
    'gat_node_classification': gat_node_classification.create_network,
    'gcn_node_classification': gcn_node_classification.create_network,
    'mlp_node_classification': mlp_node_classification.create_network,
    'gmlp_node_classification': gmlp_node_classification.create_network,
    'gmlp_graph_classification': gmlp_graph_classification.create_network,
    'gmlp_sagpool_graph_classification2': gmlp_sagpool_graph_classification.create_network2,
    'gmlp_sagpool_graph_classification3': gmlp_sagpool_graph_classification.create_network3,
    'gmlp_sagpool_graph_classification4': gmlp_sagpool_graph_classification.create_network4,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'gat_node_classification': gat_node_classification.network_modify_commandline_options,
    'gcn_node_classification': gcn_node_classification.network_modify_commandline_options,
    'mlp_node_classification': mlp_node_classification.network_modify_commandline_options,
    'gmlp_node_classification': gmlp_node_classification.network_modify_commandline_options,
    'gmlp_graph_classification': gmlp_graph_classification.network_modify_commandline_options,
    'gmlp_sagpool_graph_classification2': gmlp_sagpool_graph_classification.network_modify_commandline_options,
    'gmlp_sagpool_graph_classification3': gmlp_sagpool_graph_classification.network_modify_commandline_options,
    'gmlp_sagpool_graph_classification4': gmlp_sagpool_graph_classification.network_modify_commandline_options,
}
