import argparse
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from .cls_token_transform import create_transform as cls_token_create_transform
from .cls_token_transform import transform_modify_commandline_option as cls_token_transform_modify_commandline_option
from .indegree_transform import create_transform as indegree_create_transform
from .indegree_transform import transform_modify_commandline_option as indegree_transform_modify_commandline_option


def create_transform(opt: argparse.Namespace) -> Any:
    return Compose([cls_token_create_transform(opt), indegree_create_transform(opt)])


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = cls_token_transform_modify_commandline_option(parser)
    parser = indegree_transform_modify_commandline_option(parser)
    return parser
