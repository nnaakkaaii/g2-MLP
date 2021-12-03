import argparse
from typing import Any, Callable, Dict

from torch_geometric.transforms import Center, Compose, NormalizeScale

from . import indegree_transform, pos2attr_transform, stress_transform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'indegree': indegree_transform.create_transform,
    'pos2attr': pos2attr_transform.create_transform,
    'pos_all': lambda opt: Compose([
        Center(),
        NormalizeScale(),
        pos2attr_transform.create_transform(opt),
        indegree_transform.create_transform(opt),
    ]),
    'stress': stress_transform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'indegree': indegree_transform.transform_modify_commandline_option,
    'pos2attr': pos2attr_transform.transform_modify_commandline_option,
    'pos_all': lambda parser: pos2attr_transform.transform_modify_commandline_option(
        indegree_transform.transform_modify_commandline_option(parser)
    ),
    'stress': stress_transform.transform_modify_commandline_option,
}
