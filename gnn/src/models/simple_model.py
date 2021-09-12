import argparse
from typing import Any, Dict

import torch

from . import base_model
from .abstract_model import AbstractModel
from .activation_modules import activation_module_options, activation_modules
from .modules import module_options, modules


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

    def train_mode(self) -> None:
        super().train_mode()
        self.modules['activation_module'].is_train = True

    def eval_mode(self) -> None:
        super().eval_mode()
        self.modules['activation_module'].is_train = False

    def forward(self) -> None:
        self.h = self.modules['module'](self.x)
        self.y = self.modules['activation_module'](self.h)
