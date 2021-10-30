import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_sort_pool


class Base3LayerGNN(nn.Module):
    def __init__(self, task_type: str, dropout_rate: float):
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout_rate

        self.conv1: nn.Module = None
        self.conv2: nn.Module = None
        self.conv3: nn.Module = None
        if task_type == 'graph_classification':
            self.classifier_1: nn.Module = None
            self.classifier_2: nn.Module = None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        if hasattr(self, 'classifier_1'):
            self.classifier_1.reset_parameters()
        if hasattr(self, 'classifier_2'):
            self.classifier_2.reset_parameters()
        return

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_1 = F.dropout(F.elu(self.conv1(x, edge_index), inplace=True))
        x_2 = F.dropout(F.elu(self.conv2(x_1, edge_index) + x_1, inplace=True))
        x_3 = self.conv3(x_2, edge_index)

        if self.task_type == 'node_regression':
            return x_3.view(-1)  # ノード x 特徴量 -> flatten
        if self.task_type == 'multi_label_node_classification':
            return x_3.view(-1, 2)
        if self.task_type == 'node_classification':
            return x_3
        if self.task_type == 'graph_classification':
            out = global_sort_pool(x_3, batch, k=30)
            out = F.elu(self.classifier_1(out), inplace=True)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            out = self.classifier_2(out)
            return out.view(1, -1)

        raise NotImplementedError
