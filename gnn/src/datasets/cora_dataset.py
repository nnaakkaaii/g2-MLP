import argparse
import os
from typing import Any, Dict

import torch
import numpy as np
import scipy.sparse as sp

from . import base_dataset


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return CoraDataset(opt.max_dataset_size, opt.cora_dir, is_train, opt.train_ratio)


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--cora_dir', type=str, default='inputs/cora/', help='coraデータセットが含まれるディレクトリ')
    return parser


class CoraDataset(base_dataset.BaseDataset):
    """
    """
    idx_features_lables_filename = 'cora.content'
    edges_unordered_filename = 'cora.cites'

    def __init__(self,  max_dataset_size: int, cora_dir: str, is_train: bool, train_ratio: float) -> None:
        idx_features_labels = np.genfromtxt(os.path.join(cora_dir, self.idx_features_lables_filename), dtype=np.int32)
        edges_unordered = np.genfromtxt(os.path.join(cora_dir, self.edges_unordered_filename), dtype=np.int32)

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.__encode_onehot(idx_features_labels[:, -1])
        adj = self.__create_adj_matrix_from_idx_and_edges_unordred(idx, edges_unordered)

        features = self.__normalize_features(features)
        adj = self.__normalize_adj(adj)

        self.adj = torch.FloatTensor(np.array(adj.todense()))
        self.features = torch.FloatTensor(np.array(features.todense()))
        self.labels = torch.LongTensor(np.where(labels)[1])

        index = range(0, int(len(idx) * train_ratio)) if is_train else range(int(len(idx) * train_ratio), len(idx))
        self.index = torch.LongTensor(index)

        super().__init__(max_dataset_size, dataset_length=1, is_train=is_train)

    @staticmethod
    def __encode_onehot(labels: np.ndarray) -> np.ndarray:
        classes = sorted(list(set(labels)))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    @staticmethod
    def __create_adj_matrix_from_idx_and_edges_unordred(idx: np.ndarray, edges_unordered: np.ndarray) -> sp.coo_matrix:
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx.shape[0], idx.shape[0]))
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj

    @staticmethod
    def __normalize_adj(matrix: sp.coo_matrix) -> sp.coo_matrix:
        row_sum = np.array(matrix.sum(1))
        r_inv_sqrt = np.power(row_sum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    @staticmethod
    def __normalize_features(matrix: sp.csr_matrix) -> sp.csr_matrix:
        row_sum = np.array(matrix.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        matrix = r_mat_inv.dot(matrix)
        return matrix

    def __len__(self) -> int:
        return min(self.dataset_length, self.max_dataset_size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'index': self.index, 'features': self.features, 'labels': self.labels, 'adj': self.adj}
