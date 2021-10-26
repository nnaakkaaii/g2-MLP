import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List

import torch
import torch.nn as nn
import torchnet as tnt

from .abstract_logger import AbstractLogger


def create_logger(opt: argparse.Namespace) -> AbstractLogger:
    return BaseLogger(opt)


def logger_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--name', type=str, required=True, help='実験の固有名')
    parser.add_argument('--save_freq', type=int, default=5, help='モデルの出力の保存頻度')
    parser.add_argument('--save_dir', type=str, default=os.path.join('checkpoints'), help='モデルの出力の保存先ルートディレクトリ')
    return parser


class BaseLogger(AbstractLogger):
    """標準的なDataLoaderの実装
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__(opt)
        self.over_save_dir = os.path.join(opt.save_dir, opt.name)  # 保存先のルートディレクトリ
        self.fold_save_dir = ''  # 各foldの保存先ディレクトリ
        self.save_freq = opt.save_freq
        self._fold_number = 0

        self.loss_averager = tnt.meter.AverageValueMeter()
        self.metric_averager = tnt.meter.MSEMeter() if opt.is_regression else tnt.meter.ClassErrorMeter(accuracy=True)

        self.fold_history: DefaultDict[str, List[float]] = defaultdict(list)  # 各foldの結果
        self.over_history: DefaultDict[str, List[float]] = defaultdict(list)  # 全foldの結果

        os.makedirs(self.over_save_dir, exist_ok=True)
        self.save_options()

    def set_test_callback(self, test_callback: Callable[[], None]) -> None:
        self._test_callback = test_callback
        return

    def set_network(self, network: nn.Module) -> None:
        self._network = network
        return

    def on_start(self, state: Dict[str, Any]) -> None:
        if state['training']:
            self.fold_save_dir = os.path.join(self.over_save_dir, f'{self._fold_number:02}')
            self.fold_history = defaultdict(list)
            self._fold_number += 1

            os.makedirs(self.fold_save_dir, exist_ok=True)
        return

    def on_end(self, state: Dict[str, Any]) -> None:
        if state['training']:
            for key, value in self.fold_history.items():
                self.over_history[key].append(value[-1])  # 最後の値をover_historyに保存

            with open(os.path.join(self.fold_save_dir, 'history.json'), 'w') as f:
                json.dump(self.fold_history, f)
        return

    def on_sample(self, state: Dict[str, Any]) -> None:
        state['sample'] = state['sample'], state['train']
        return

    def on_forward(self, state: Dict[str, Any]) -> None:
        self.loss_averager.add(state['loss'].detach().cpu().item())
        self.metric_averager.add(state['output'].detach().cpu(), state['sample'][0].y)
        self.print_status(state)
        return

    def on_end_all_training(self) -> None:
        with open(os.path.join(self.over_save_dir, 'history.json'), 'w') as f:
            json.dump(self.over_history, f)
        return

    def _reset_averager(self) -> None:
        self.loss_averager.reset()
        self.metric_averager.reset()
        return

    def on_start_epoch(self, state: Dict[str, Any]) -> None:
        self._reset_averager()
        return

    def on_end_epoch(self, state: Dict[str, Any]) -> None:
        """trainでのみ呼ばれる
        """
        epoch = state['epoch']
        self.fold_history['train_loss'].append(self.loss_averager.value()[0])
        self.fold_history['train_metric'].append(self.metric_averager.value()[0])

        if self._test_callback is not None:
            self._reset_averager()
            with torch.no_grad():
                self._test_callback()

            self.fold_history['test_loss'].append(self.loss_averager.value()[0])
            self.fold_history['test_metric'].append(self.metric_averager.value()[0])

        if epoch % self.save_freq == 0 and self._network is not None:
            self.save_models(state)
        return

    def save_models(self, state: Dict[str, Any]) -> None:
        assert state['train'] and self._network is not None
        epoch = state['epoch']
        save_path = os.path.join(self.fold_save_dir, f'net_{epoch}.pth')
        torch.save(self._network.state_dict(), save_path)
        return

    def save_options(self) -> None:
        with open(os.path.join(self.over_save_dir, 'options.json'), 'w') as f:
            json.dump(vars(self.opt), f)
        return

    def print_status(self, state: Dict[str, Any]) -> None:
        iteration = state['t']
        max_iteration = len(state['iterator'])
        if state['train']:
            epoch = state['epoch']
            status_str = f'[Epoch {epoch}][{iteration:.0f}/{max_iteration}] '
            status_str += f'train_loss: {self.loss_averager.value()[0]:.4f}, '
            status_str += f'train_metric: {self.metric_averager.value()[0]:.4f}, '
        else:
            status_str = f'[{iteration:.0f}/{max_iteration}] '
            status_str += f'test_loss: {self.loss_averager.value()[0]:.4f}, '
            status_str += f'test_metric: {self.metric_averager.value()[0]:.4f}, '

        sys.stdout.write(f'\r{status_str}')
        sys.stdout.flush()
        return

    def print_networks(self) -> None:
        assert self._network is not None
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self._network.parameters():
            num_params += param.numel()
        print(self._network)
        print(f'Total number of parameters: {num_params / 1e6:.3f} M')
        print('-----------------------------------------------')
        return
