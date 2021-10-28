import argparse
import os

import torch

from ..datasets import dataset_options, datasets
from ..models.losses import loss_options, losses
from ..models.networks import network_options, networks


class BaseOption:
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions
        in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized.
        """
        self.initialized = False
        self.is_train: bool = True  # overwrite
        self.parser = None
        self.opt = None

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Define the common options that are used in both training and test.
        """
        parser.add_argument('--name', type=str, required=True, help='実験の固有名')
        parser.add_argument('--mlflow_root_dir', type=str, default=os.path.join('mlruns'))
        parser.add_argument('--run_name', type=str, default='test')
        parser.add_argument('--save_freq', type=int, default=5, help='モデルの出力の保存頻度')
        parser.add_argument('--save_dir', type=str, default=os.path.join('checkpoints'), help='モデルの出力の保存先ルートディレクトリ')
        parser.add_argument('--no_visdom_logger', action='store_true', help='visdomのloggerを利用しない')

        parser.add_argument('--gpu_ids', type=str, default='0', help='使用するGPUのIDをカンマ区切り')
        parser.add_argument('--verbose', action='store_true', help='詳細を表示するか')

        parser.add_argument('--loss_name', type=str, required=True, choices=losses.keys())
        parser.add_argument('--network_name', type=str, required=True, choices=networks.keys())
        parser.add_argument('--dataset_name', type=str, required=True, choices=datasets.keys())

        parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')

        parser.add_argument('--task_type', type=str, required=True,
                            choices=['node_classification', 'multi_label_node_classification', 'node_regression', 'graph_classification'])

        self.initialized = True
        return parser

    def gather_options(self) -> argparse.ArgumentParser:
        """Initialize our parser with basic options only once.
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function in model and dataset class
        """
        if self.initialized:
            assert False, 'option class has already been initialized.'

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()  # extract arguments; modify following arguments dynamically
        loss_modify_commandline_options = loss_options[opt.loss_name]
        parser = loss_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        network_modify_commandline_options = network_options[opt.network_name]
        parser = network_modify_commandline_options(parser)

        opt, _ = parser.parse_known_args()
        dataset_modify_commandline_options = dataset_options[opt.dataset_name]
        parser = dataset_modify_commandline_options(parser)

        self.parser = parser

        return parser

    def print_options(self, opt: argparse.Namespace) -> None:
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [save_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.save_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        filename = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(filename, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        return

    def parse(self) -> argparse.Namespace:
        """Parse our options, and set up gpu device.
        :return:
        """
        opt = self.gather_options().parse_args()
        opt.is_train = self.is_train
        opt.phase = 'train' if self.is_train else 'test'

        # set gpu ids
        opt.gpu_ids = str(opt.gpu_ids)  # 型ヒントの回避
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id_ = int(str_id)
            if id_ >= 0:
                opt.gpu_ids.append(id_)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        if opt.verbose:
            self.print_options(opt)

        self.opt = opt
        return opt
