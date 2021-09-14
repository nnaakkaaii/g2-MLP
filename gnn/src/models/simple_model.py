import argparse
from typing import Any, Dict

import torch

from . import base_model
from .abstract_model import AbstractModel
from .modules import module_options, modules
from .modules.activation_modules import (activation_module_options,
                                         activation_modules)


def create_model(opt: argparse.Namespace) -> AbstractModel:
    return SimpleModel(opt)


def model_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_model.model_modify_commandline_options(parser)

    parser.add_argument('--module_name', type=str, required=True, choices=modules.keys())
    opt, _ = parser.parse_known_args()
    module_modify_command_line_options = module_options[opt.module_name]
    parser = module_modify_command_line_options(parser)

    parser.add_argument('--activation_module_name', type=str, required=True, choices=activation_modules.keys())
    opt, _ = parser.parse_known_args()
    activation_module_modify_command_line_options = activation_module_options[opt.activation_module_name]
    parser = activation_module_modify_command_line_options(parser)

    # 画像の入力に必要なオプション
    parser.add_argument('--in_size', type=int, default=28, choices=[28], help='入力画像の大きさ. dimはこの2乗に')
    parser.add_argument('--in_nch', type=int, default=1, choices=[1, 3], help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--out_dim', type=int, default=10, help='分類のクラス数')
    # moduleで共通に利用するオプション
    parser.add_argument('--last_p_dim', type=int, default=256, help='最終層のパーセプトロン数')

    return parser


class SimpleModel(base_model.BaseModel):
    def __init__(self, opt: argparse.Namespace) -> None:
        self.modules = {
            'module': modules[opt.module_name](opt),
            'activation_module': activation_modules[opt.activation_module_name](opt),
        }
        super().__init__(opt)
        self.h = torch.tensor(0)

    def set_input(self, input_: Dict[str, Any]) -> None:
        super().set_input(input_)
        self.modules['activation_module'].label = self.t

    def forward(self) -> None:
        self.h = self.modules['module'](self.x)
        self.y = self.modules['activation_module'](self.h)
