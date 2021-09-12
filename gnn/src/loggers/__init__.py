import argparse
from typing import Callable, Dict

from ..models.abstract_model import AbstractModel
from . import mlflow_logger, simple_logger
from .abstract_logger import AbstractLogger

loggers: Dict[str, Callable[[AbstractModel, argparse.Namespace], AbstractLogger]] = {
    'simple': simple_logger.create_logger,
    'mlflow': mlflow_logger.create_logger,
}

logger_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_logger.logger_modify_commandline_options,
    'mlflow': mlflow_logger.logger_modify_commandline_options,
}
