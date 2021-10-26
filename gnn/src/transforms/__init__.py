import argparse
from typing import Any, Callable, Dict

from . import indegree_tranform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'indegree': indegree_tranform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'indegree': indegree_tranform.transform_modify_commandline_option,
}
