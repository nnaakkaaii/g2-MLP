import argparse
from typing import Any, Callable, Dict

from . import cosine_scheduler, linear_scheduler, step_scheduler

schedulers: Dict[str, Callable[[Any, argparse.Namespace], Any]] = {
    'cosine': cosine_scheduler.create_scheduler,
    'linear': linear_scheduler.create_scheduler,
    'step': step_scheduler.create_scheduler,
}

scheduler_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'cosine': lambda x: x,
    'linear': linear_scheduler.modify_scheduler_commandline_options,
    'step': step_scheduler.modify_scheduler_commandline_options,
}
