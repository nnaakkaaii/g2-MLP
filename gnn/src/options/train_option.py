import argparse

from .base_option import BaseOption
from ..transforms import transforms, transform_options


class TrainOption(BaseOption):
    """This class includes training options.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.add_argument('--train_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--val_transform_name', type=str, required=True, choices=transforms.keys())
        # training parameters
        parser.add_argument('--train_ratio', type=float, default=0.8, help='学習データと検証データに占める学習データの割合')
        parser.add_argument('--n_epochs', type=int, default=100, help='初期LearningRateで実行するエポック数')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='LearningRateを減衰させながら実行するエポック数')
        parser.add_argument('--epoch', type=int, default=1, help='事前学習時の最後のepoch (読み込みたい重みのepoch)')
        parser.add_argument('--continue_train', action='store_true', help='前回の学習を続行するか')

        self.is_train = True
        return parser

    def gather_options(self) -> argparse.ArgumentParser:
        parser = super().gather_options()

        opt, _ = parser.parse_known_args()
        train_transform_modify_commandline_options = transform_options[opt.train_transform_name]
        parser = train_transform_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        val_transform_modify_commandline_options = transform_options[opt.val_transform_name]
        parser = val_transform_modify_commandline_options(parser)

        self.parser = parser
        return parser
