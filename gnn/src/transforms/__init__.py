import argparse
from typing import Any, Callable, Dict

from torch_geometric.transforms import Center, Compose, NormalizeScale

from . import (disp_as_label_transform, indegree_transform,
               pos_as_attr_transform, stress_as_label_transform,
               stress_cls_transform)

# disp_as_label, stress_as_label, stress_cls は pre_transform としてしか使わない
transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'indegree': indegree_transform.create_transform,
    'pos_as_attr': lambda opt: Compose([  # clsのtransformに用いる. posをnode attrにするのみ
        Center(),
        NormalizeScale(),
        pos_as_attr_transform.create_transform(opt),
        indegree_transform.create_transform(opt),
    ]),
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'indegree': indegree_transform.transform_modify_commandline_option,
    'pos_as_attr': lambda parser: pos_as_attr_transform.transform_modify_commandline_option(  # clsのtransformに用いる. posをnode attrにするのみ
        indegree_transform.transform_modify_commandline_option(parser)
    ),
}
