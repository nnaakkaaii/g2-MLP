import argparse
from typing import Callable, Dict

from . import base_logger, mlflow_logger, visdom_plot_logger
from .abstract_logger import AbstractLogger

loggers: Dict[str, Callable[[argparse.Namespace], AbstractLogger]] = {
    'base': base_logger.create_logger,
    'mlflow': mlflow_logger.create_logger,
    'visdom_plot': visdom_plot_logger.create_logger,
}

logger_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'base': base_logger.logger_modify_commandline_options,
    'mlflow': mlflow_logger.logger_modify_commandline_options,
    'visdom_plot': visdom_plot_logger.logger_modify_commandline_options,
}
