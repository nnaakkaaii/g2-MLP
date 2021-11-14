import argparse
from typing import Any, Callable, Dict

from . import indegree_transform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'indegree': indegree_transform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'indegree': indegree_transform.transform_modify_commandline_option,
}
