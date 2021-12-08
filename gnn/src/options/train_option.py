import argparse
import os

from ..models.losses import loss_options, losses
from ..models.optimizers import optimizer_options, optimizers
from ..models.schedulers import scheduler_options, schedulers
from ..transforms import transform_options, transforms
from .base_option import BaseOption


class TrainOption(BaseOption):
    """This class includes training options.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)

        parser.add_argument('--loss_name', type=str, required=True, choices=losses.keys())
        parser.add_argument('--train_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--val_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--optimizer_name', type=str, required=True, choices=optimizers.keys())
        parser.add_argument('--scheduler_name', type=str, required=True, choices=schedulers.keys())

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='初期LearningRateで実行するエポック数')

        self.is_train = True
        return parser

    def gather_options(self) -> argparse.ArgumentParser:
        parser = super().gather_options()

        opt, _ = parser.parse_known_args()  # extract arguments; modify following arguments dynamically
        loss_modify_commandline_options = loss_options[opt.loss_name]
        parser = loss_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        train_transform_modify_commandline_options = transform_options[opt.train_transform_name]
        parser = train_transform_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        val_transform_modify_commandline_options = transform_options[opt.val_transform_name]
        parser = val_transform_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        optimizer_modify_commandline_options = optimizer_options[opt.optimizer_name]
        parser = optimizer_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        scheduler_modify_commandline_options = scheduler_options[opt.scheduler_name]
        parser = scheduler_modify_commandline_options(parser)

        self.parser = parser
        return parser
