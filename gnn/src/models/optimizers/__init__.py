import argparse
from typing import Any, Callable, Dict

from . import adam_optimizer

optimizers: Dict[str, Callable[[Any, argparse.Namespace], Any]] = {
    'adam': adam_optimizer.create_optimizer,
}

optimizer_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'adam': adam_optimizer.optimizer_modify_commandline_options,
}
