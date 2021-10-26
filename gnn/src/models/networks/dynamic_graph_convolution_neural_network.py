import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Dropout, Linear, MaxPool1d
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import remove_self_loops


def create_network(opt: argparse.Namespace) -> nn.Module:
    return DGCNN(opt.num_features, opt.num_classes, opt.is_regression)


def network_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--num_features', type=int, required=True, help='特徴量の数')
    parser.add_argument('--num_classes', type=int, default=1, help='予測対象のクラス数')
    return parser


class DGCNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int, is_regression: bool) -> None:
        super().__init__()
        self.is_regression = is_regression

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(1, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(352, 128)
        self.drop_out = Dropout(0.5)
        if not is_regression:
            self.classifier_2 = Linear(128, num_classes)
        else:
            self.classifier_2 = Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)

        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch, k=30)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        out = self.classifier_2(out)
        if not self.is_regression:
            out = F.log_softmax(self.classifier_2(out), dim=-1)

        return out