import argparse
from typing import Callable, Dict

from .abstract_model import AbstractModel

from . import simple_model

models: Dict[str, Callable[[argparse.Namespace], AbstractModel]] = {
    'simple': simple_model.create_model,
}

model_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_model.model_modify_commandline_options,
}
