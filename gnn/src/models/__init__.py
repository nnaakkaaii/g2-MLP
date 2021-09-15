import argparse
from typing import Callable, Dict

from . import graph_attention_model, simple_model
from .abstract_model import AbstractModel

models: Dict[str, Callable[[argparse.Namespace], AbstractModel]] = {
    'simple': simple_model.create_model,
    'graph_attention_model': graph_attention_model.create_model,
}

model_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_model.model_modify_commandline_options,
    'graph_attention_model': graph_attention_model.model_modify_commandline_options,
}
