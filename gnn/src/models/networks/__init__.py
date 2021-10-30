import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import gat, gcn, ggat, ggatpool, gnn_sagpool

networks: Dict[str, Callable[[int, int, argparse.Namespace], nn.Module]] = {
    'GAT': gat.create_network,
    'GCN': gcn.create_network,
    'GGAT': ggat.create_network,
    'GGATPool': ggatpool.create_network,
    'GNNSAGPool': gnn_sagpool.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'GAT': gat.network_modify_commandline_options,
    'GCN': gcn.network_modify_commandline_options,
    'GGAT': ggat.network_modify_commandline_options,
    'GGATPool': ggatpool.network_modify_commandline_options,
    'GNNSAGPool': gnn_sagpool.network_modify_commandline_options,
}
