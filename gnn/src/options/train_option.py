import argparse

from ..models.optimizers import optimizer_options, optimizers
from ..transforms import transform_options, transforms
from .base_option import BaseOption


class TrainOption(BaseOption):
    """This class includes training options.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.add_argument('--train_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--test_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--optimizer_name', type=str, required=True, choices=optimizers.keys())

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='初期LearningRateで実行するエポック数')

        self.is_train = True
        return parser

    def gather_options(self) -> argparse.ArgumentParser:
        parser = super().gather_options()

        opt, _ = parser.parse_known_args()
        train_transform_modify_commandline_options = transform_options[opt.train_transform_name]
        parser = train_transform_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        test_transform_modify_commandline_options = transform_options[opt.test_transform_name]
        parser = test_transform_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        optimizer_modify_commandline_options = optimizer_options[opt.optimizer_name]
        parser = optimizer_modify_commandline_options(parser)

        self.parser = parser
        return parser
