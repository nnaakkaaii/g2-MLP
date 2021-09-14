import argparse
from typing import Dict, Any

import numpy as np

from . import base_model
from .abstract_model import AbstractModel
from .modules import module_options, modules
from .utils.metrics import get_metrics


def create_model(opt: argparse.Namespace) -> AbstractModel:
    return GraphAttentionModel(opt)


def model_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_model.model_modify_commandline_options(parser)

    parser.add_argument('--module_name', type=str, required=True, choices=modules.keys())
    opt, _ = parser.parse_known_args()
    module_modify_commandline_options = module_options[opt.module_name]
    parser = module_modify_commandline_options(parser)

    # グラフの入力に必要なオプション
    parser.add_argument('--in_dim', type=int, default=2708, help='入力グラフのノード数')
    parser.add_argument('--out_dim', type=int, default=7, help='分類のクラス数')

    return parser


class GraphAttentionModel(base_model.BaseModel):
    def __init__(self, opt: argparse.Namespace) -> None:
        self.modules = {
            'module': modules[opt.module_name](opt),
        }
        super().__init__(opt)

    def set_input(self, input_: Dict[str, Any]) -> None:
        self.x = input_['features']
        self.t = input_['labels']
        self.mask = input_['adj']
        self.index = input_['index']

    def forward(self) -> None:
        self.y = self.modules['module'](self.x, self.mask)

    def _calc_loss_and_metrics(self) -> None:
        self.loss = self.criterion(self.y[self.index], self.t[self.index])
        y_true = self.t.detach().clone().cpu().numpy()
        y_pred = np.argmax(self.y.detach().clone().cpu().numpy(), axis=1)
        self.metrics = get_metrics(y_true, y_pred)
