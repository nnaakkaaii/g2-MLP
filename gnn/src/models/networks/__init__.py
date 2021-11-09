import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import gat
from . import gcn

networks: Dict[str, Callable[[int, int, bool, argparse.Namespace], nn.Module]] = {
    'gat': gat.create_network,
    'gcn': gcn.create_network,
}

network_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'gat': gat.network_modify_commandline_options,
    'gcn': gcn.network_modify_commandline_options,
}
