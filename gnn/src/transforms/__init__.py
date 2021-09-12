import argparse
from typing import Any, Callable, Dict

from . import affine_transform, no_transform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'no': no_transform.create_transform,
    'affine': affine_transform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'no': lambda x: x,
    'affine': affine_transform.transform_modify_commandline_option,
}
