import argparse

from .base_option import BaseOption
from ..transforms import transforms, transform_options


class TestOption(BaseOption):
    """This class includes test options.
    It also includes shared options defined in BaseOption.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.add_argument('--test_transform_name', type=str, required=True, choices=transforms.keys())
        parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        self.is_train = False
        return parser

    def gather_options(self) -> argparse.ArgumentParser:
        parser = super().gather_options()

        opt, _ = parser.parse_known_args()
        test_transform_modify_commandline_options = transform_options[opt.test_transform_name]
        parser = test_transform_modify_commandline_options(parser)

        self.parser = parser
        return parser
