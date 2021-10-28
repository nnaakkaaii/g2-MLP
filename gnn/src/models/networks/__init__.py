import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import graph_attention_network, graph_convolution_network

networks: Dict[str, Callable[[int, int, argparse.Namespace], nn.Module]] = {
    'GAT': graph_attention_network.create_network,
    'GCN': graph_convolution_network.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'GAT': graph_attention_network.network_modify_commandline_options,
    'GCN': graph_convolution_network.network_modify_commandline_options,
}
