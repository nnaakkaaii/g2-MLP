import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import gat, gcn, ggat, ggat_unet, ggatpool, gnn_pool, unet

networks: Dict[str, Callable[[int, int, argparse.Namespace], nn.Module]] = {
    'GAT': gat.create_network,
    'GCN': gcn.create_network,
    'GGAT': ggat.create_network,
    'GGATPool': ggatpool.create_network,
    'GNNPool': gnn_pool.create_network,
    'GGATUNet': ggat_unet.create_network,
    'UNet': unet.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'GAT': gat.network_modify_commandline_options,
    'GCN': gcn.network_modify_commandline_options,
    'GGAT': ggat.network_modify_commandline_options,
    'GGATPool': ggatpool.network_modify_commandline_options,
    'GNNPool': gnn_pool.network_modify_commandline_options,
    'GGATUNet': ggat_unet.network_modify_commandline_options,
    'UNet': unet.network_modify_commandline_options,
}
