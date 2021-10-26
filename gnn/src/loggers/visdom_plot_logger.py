import argparse
from typing import Any, Dict

import torch
from torchnet import logger

from . import base_logger
from .abstract_logger import AbstractLogger


def create_logger(opt: argparse.Namespace) -> AbstractLogger:
    return VisdomPlotLogger(opt)


def logger_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_logger.logger_modify_commandline_options(parser)
    return parser


class VisdomPlotLogger(base_logger.BaseLogger):
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__(opt)
        self.train_loss_logger = logger.VisdomPlotLogger('line', env=opt.name, opts={'title': 'Train Loss'})
        self.train_metric_logger = logger.VisdomPlotLogger('line', env=opt.name, opts={'title': 'Train Metric'})
        self.test_loss_logger = logger.VisdomPlotLogger('line', env=opt.name, opts={'title': 'Test Loss'})
        self.test_metric_logger = logger.VisdomPlotLogger('line', env=opt.name, opts={'title': 'Test Metric'})

    def on_end(self, state: Dict[str, Any]) -> None:
        if state['train']:
            self.test_loss_logger.log()

    def on_end_epoch(self, state: Dict[str, Any]) -> None:
        epoch = state['epoch']
        self.fold_history['train_loss'].append(self.loss_averager.value()[0])
        self.fold_history['train_metric'].append(self.metric_averager.value()[0])
        self.train_loss_logger.log(state['epoch'], self.loss_averager.value()[0], name='fold_' + str(self._fold_number))
        self.train_metric_logger.log(state['epoch'], self.metric_averager.value()[0], name='fold_' + str(self._fold_number))

        if self._test_callback is not None:
            self._reset_averager()
            with torch.no_grad():
                self._test_callback()

            self.fold_history['test_loss'].append(self.loss_averager.value()[0])
            self.fold_history['test_metric'].append(self.metric_averager.value()[0])
            self.test_loss_logger.log(state['epoch'], self.loss_averager.value()[0], name='fold_' + str(self._fold_number))
            self.test_metric_logger.log(state['epoch'], self.metric_averager.value()[0], name='fold_' + str(self._fold_number))

        if epoch % self.save_freq == 0 and self._network is not None:
            self.save_models(state)
        return
