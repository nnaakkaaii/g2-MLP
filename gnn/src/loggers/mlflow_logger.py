import argparse
import os

import mlflow

from ..models.base_model import AbstractModel
from . import simple_logger
from .abstract_logger import AbstractLogger


def create_logger(model: AbstractModel, opt: argparse.Namespace) -> AbstractLogger:
    return MLflowLogger(model, opt)


def logger_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = simple_logger.logger_modify_commandline_options(parser)
    parser.add_argument('--mlflow_root_dir', type=str, default=os.path.join('mlruns'))
    parser.add_argument('--run_name', type=str, default='test')
    return parser


class MLflowLogger(simple_logger.SimpleLogger):
    def __init__(self, model: AbstractModel, opt: argparse.Namespace) -> None:
        super().__init__(model=model, opt=opt)
        self.__initialize_mlflow_logger()

    def __initialize_mlflow_logger(self) -> None:
        mlflow_root_dir = self.opt['mlflow_root_dir']
        os.makedirs(mlflow_root_dir, exist_ok=True)

        mlflow.set_tracking_uri(mlflow_root_dir)

        task_name = self.opt['name']
        run_name = self.opt['run_name']

        if not bool(mlflow.get_experiment_by_name(task_name)):
            mlflow.create_experiment(task_name, artifact_location=None)

        mlflow.set_experiment(task_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(self.opt)
        return

    def end_epoch(self) -> None:
        super().end_epoch()
        mlflow.log_metrics(self.train_averager.value(), step=self._trained_epoch)
        mlflow.log_metrics(self.val_averager.value(), step=self._trained_epoch)
        return

    def end_all_training(self) -> None:
        super().end_all_training()
        mlflow.end_run()
        return
