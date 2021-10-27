import argparse

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.loader import DataLoader


def create_dataloader(dataset: InMemoryDataset, is_train: bool, opt: argparse.Namespace) -> DataLoader:
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches and is_train, num_workers=int(opt.num_threads))


def dataloader_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--serial_batches', action='store_true', help='dataloaderの読み込み順をランダムにするか')
    parser.add_argument('--num_threads', type=int, default=0, help='データローダーの並列数')
    return parser
