import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import dynamic_graph_convolution_neural_network

networks: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'DGCNN': dynamic_graph_convolution_neural_network.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'DGCNN': dynamic_graph_convolution_neural_network.network_modify_commandline_options,
}
