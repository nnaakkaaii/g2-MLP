import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import (gat_node, gcn_node, gmlp_graph, gmlp_hierarchical_sagpool_graph,
               gmlp_node, mlp_node)

networks: Dict[str, Callable[[int, int, bool, argparse.Namespace], nn.Module]] = {
    'gat_node': gat_node.create_network,
    'gcn_node': gcn_node.create_network,
    'gmlp_node': gmlp_node.create_network,
    'gmlp_graph': gmlp_graph.create_network,
    'gmlp_hierarchical_sagpool_graph': gmlp_hierarchical_sagpool_graph.create_network,
    'mlp_node': mlp_node.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'gat_node': gat_node.network_modify_commandline_options,
    'gcn_node': gcn_node.network_modify_commandline_options,
    'gmlp_node': gmlp_node.network_modify_commandline_options,
    'gmlp_graph': gmlp_graph.network_modify_commandline_options,
    'gmlp_hierarchical_sagpool_graph': gmlp_hierarchical_sagpool_graph.network_modify_commandline_options,
    'mlp_node': mlp_node.network_modify_commandline_options,
}
