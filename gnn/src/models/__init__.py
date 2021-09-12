import argparse
from typing import Callable, Dict

from . import simple_model
from .abstract_model import AbstractModel

models: Dict[str, Callable[[argparse.Namespace], AbstractModel]] = {
    'simple': simple_model.create_model,
}

model_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_model.model_modify_commandline_options,
}
