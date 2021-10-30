from typing import Dict

import torch.nn as nn
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn.pool import TopKPooling

from .ggat_module import AbstractGGATBlock, GGAT1Block, GGAT2Block
from .sagpool_module import SAGPooling

GGAT_TYPES: Dict[str, AbstractGGATBlock] = {
    'GGAT1': GGAT1Block,
    'GGAT2': GGAT2Block,
}

GNN_TYPES: Dict[str, nn.Module] = {
    'GCN': GCNConv,
    'GAT': GATConv,
}

POOL_TYPES: Dict[str, nn.Module] = {
    'TopKPool': TopKPooling,
    'SAGPool': SAGPooling,
}
