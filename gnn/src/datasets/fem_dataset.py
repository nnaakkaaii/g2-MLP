import argparse
import os
from itertools import product
from typing import Any

import numpy as np
import requests
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip

from src.transforms.stress_transform import create_transform


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> InMemoryDataset:
    dataset = FemDataset(opt.data_dir, split='train' if is_train else 'test', transform=transform, pre_transform=create_transform(opt))
    return dataset


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data_dir', type=str, default=os.path.join('inputs', 'FEM'), help='FEMデータを保存する場所')
    return parser


def download_gdrive_file(id: str, save_path: str) -> None:
    session = requests.session()
    res = session.get(f'https://drive.google.com/uc?export=download&id={id}')
    cookie = [v for k, v in res.cookies.items() if '_warning_' in k][0]
    res = session.get(f'https://drive.google.com/uc?export=download&confirm={cookie}&id={id}', allow_redirects=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(res.content)
    return


class FemDataset(InMemoryDataset):

    id = '1-5-GZWaL1cqoTzPysv--qHOAmVqX8Opx'

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']

        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        splits = ['train', 'val', 'test']
        files = ['x.npz', 'edge_index.npz', 'pos.npz', 'y.npz']
        return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = os.path.join(self.raw_dir, 'fem.zip')
        download_gdrive_file(self.id, save_path=path)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        return

    def process(self):
        data_list = []

        x = np.load(os.path.join(self.raw_dir, f'{self.split}_x.npz'))
        pos = np.load(os.path.join(self.raw_dir, f'{self.split}_pos.npz'))
        edge_index = np.load(os.path.join(self.raw_dir, f'{self.split}_edge_index.npz'))
        y = np.load(os.path.join(self.raw_dir, f'{self.split}_y.npz'))

        for key in x.keys():
            data = Data(
                x=torch.from_numpy(x[key]),
                edge_index=torch.from_numpy(edge_index[key]),
                y=torch.from_numpy(y[key]),
                pos=torch.from_numpy(pos[key]),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, f'{self.split}.pt'))
        return
