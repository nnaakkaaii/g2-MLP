import argparse
from typing import Any, Callable, Dict

from . import cls_token_transform, indegree_transform, cls_token_indegree_transform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'indegree': indegree_transform.create_transform,
    'cls_token': cls_token_transform.create_transform,
    'cls_token_indegree': cls_token_indegree_transform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'indegree': indegree_transform.transform_modify_commandline_option,
    'cls_token': cls_token_transform.transform_modify_commandline_option,
    'cls_token_indegree': cls_token_indegree_transform.transform_modify_commandline_option,
}
