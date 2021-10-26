import argparse
import os
from typing import Any, Dict

import mlflow

from . import base_logger
from .abstract_logger import AbstractLogger


def create_logger(opt: argparse.Namespace) -> AbstractLogger:
    return MLflowLogger(opt)


def logger_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_logger.logger_modify_commandline_options(parser)
    parser.add_argument('--mlflow_root_dir', type=str, default=os.path.join('mlruns'))
    parser.add_argument('--run_name', type=str, default='test')
    return parser


class MLflowLogger(base_logger.BaseLogger):
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__(opt)

        os.makedirs(opt.mlflow_root_dir, exist_ok=True)
        mlflow.set_tracking_uri(opt.mlflow_root_dir)

        if not bool(mlflow.get_experiment_by_name(opt.name)):
            mlflow.create_experiment(opt.name, artifact_location=None)

        mlflow.set_experiment(opt.name)
        mlflow.start_run(run_name=opt.run_name)
        mlflow.log_params(vars(self.opt))
        return

    def on_end_epoch(self, state: Dict[str, Any]) -> None:
        super().on_end_epoch(state)
        metrics = {
            'train_loss': self.fold_history['train_loss'][-1],
            'train_metric': self.fold_history['train_metric'][-1],
        }
        if self._test_callback is not None:
            metrics.update({
                'test_loss': self.fold_history['test_loss'][-1],
                'test_metric': self.fold_history['test_metric'][-1],
            })
        mlflow.log_metrics(metrics, step=state['epoch'])
        return

    def on_end_all_training(self) -> None:
        super().on_end_all_training()
        mlflow.end_run()
