import abc
import argparse
import os
from typing import Any, Dict, List, Union

import torch
import numpy as np

from .abstract_model import AbstractModel
from .losses import losses, loss_options
from .optimizers import optimizer_options, optimizers
from .schedulers import scheduler_options, schedulers
from .utils.init_weights import init_weight_options, init_weights
from .utils.metrics import get_metrics


def model_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add new model-specific options, and rewrite default values for existing options.
    modulesに含むオプションは設定していないので、継承後のクラスとともに実装する
    """
    # criterion
    parser.add_argument('--loss_name', type=str, required=True, choices=losses.keys())
    opt, _ = parser.parse_known_args()
    loss_modify_command_line_options = loss_options[opt.loss_name]
    parser = loss_modify_command_line_options(parser)

    # optimizer
    parser.add_argument('--optimizer_name', type=str, required=True, choices=optimizers.keys())
    opt, _ = parser.parse_known_args()
    optimizer_modify_commandline_options = optimizer_options[opt.optimizer_name]
    parser = optimizer_modify_commandline_options(parser)

    # scheduler
    parser.add_argument('--scheduler_name', type=str, required=True, choices=schedulers.keys())
    opt, _ = parser.parse_known_args()
    scheduler_modify_commandline_options = scheduler_options[opt.scheduler_name]
    parser = scheduler_modify_commandline_options(parser)

    # init weight
    parser.add_argument('--init_weight_name', type=str, required=True, choices=init_weights.keys())
    opt, _ = parser.parse_known_args()
    init_weight_modify_commandline_options = init_weight_options[opt.init_weight_name]
    parser = init_weight_modify_commandline_options(parser)

    return parser


class BaseModel(AbstractModel, metaclass=abc.ABCMeta):
    """This class is an abstract class for models.
    # base_modelで行うこと
    set_inputでは、input_にx,tのキーが存在していることを前提とする
    _calc_loss_and_metricsではlossと基本のmetrics計算を行う
    backwardで_calc_loss_and_metricsの呼び出しとlossのbackward計算をする
    # 継承先で行うべきこと
    forwardの実装 (その中でself.yの計算)
    modulesの上書き
    (任意)optimizerやschedulerの上書き(lrを変更する場合)
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialize the BaseModel class.
        """
        super().__init__(opt)

        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train  # activation moduleなどに学習状況を渡すために状態保持
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.loss = torch.tensor(0)  # criterionより計算したロス
        self.metrics: Dict[str, float] = {}  # _calc_loss_and_metricsで計算

        self.x: torch.Tensor = torch.tensor(0)
        self.y: torch.Tensor = torch.tensor(0)
        self.t: torch.Tensor = torch.tensor(0)

        # moduleはmodelごとに定義
        if self.is_train:
            self.criterion = losses[opt.loss_name](opt)
            params = [{'params': v.parameters()} for v in self.modules.values()]  # lrを個別に設定する場合は継承後のクラスで設定
            self.optimizer = optimizers[opt.optimizer_name](params, opt)
            self.scheduler = schedulers[opt.scheduler_name](self.optimizer, opt)

        self.module_transfer_to_device()

    def module_transfer_to_device(self) -> None:
        """transfer all modules to device
        """
        for key, module in self.modules.items():
            module.to(self.device)
            if self.device.type == 'cuda':
                self.modules[key] = torch.nn.DataParallel(module, self.gpu_ids)
        return

    def setup(self, opt: argparse.Namespace) -> None:
        """Called in construct
        Setup (Load and print networks)
            -- load networks    : if not training mode or continue_train is True, then load opt.epoch
            -- print networks
        """
        if not self.is_train or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)
        return

    def set_input(self, input_: Dict[str, Any]) -> None:
        self.x = input_['x'].to(self.device)
        self.t = input_['t'].to(self.device)
        return

    def backward(self) -> None:
        """calc loss and metrics, and backward loss
        """
        self._calc_loss_and_metrics()
        self.loss.backward()
        return

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights; called in every training iteration.
        iterationごとの学習の1ループ
        """
        self.forward()
        # update discriminator
        self.optimizer.zero_grad()                       # set gradients to zero
        self.backward()                                   # calculate gradients
        self.optimizer.step()                            # update weights
        return

    def test(self) -> None:
        """ Forward function used in test time.
        iterationごとの検証の1ループ
        """
        with torch.no_grad():
            self.forward()
            self._calc_loss_and_metrics()
        return

    def train_mode(self) -> None:
        """turn to train mode
        """
        self.is_train = True
        for module in self.modules.values():
            module.train()
        return

    def eval_mode(self) -> None:
        """make models eval mode during test time.
        """
        self.is_train = False
        for module in self.modules.values():
            module.eval()
        return

    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks; called at the end of every epoch
        """
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
        return

    def _calc_loss_and_metrics(self) -> None:
        self.loss = self.criterion(self.y, self.t)
        y_true = self.t.detach().clone().cpu().numpy()
        y_pred = np.argmax(self.y.detach().clone().cpu().numpy(), axis=1)
        self.metrics = get_metrics(y_true, y_pred)

    def get_current_loss_and_metrics(self) -> Dict[str, float]:
        """Return training losses / errors. train_option.py will print out these errors on console, and save them to a file
        trainもvalもlossとmetricsはこのインターフェイスから取得する
        """
        errors_ret = {'loss': float(self.loss.detach().clone().cpu().numpy())}
        errors_ret.update(self.metrics)
        return errors_ret

    def save_networks(self, epoch: Union[int, str]) -> None:
        """Save all the networks to the disk.
        """
        for name, module in self.modules.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(module.module.cpu().state_dict(), save_path)
                module.cuda(self.gpu_ids[0])
            else:
                torch.save(module.cpu().state_dict(), save_path)
        return

    def __patch_instance_norm_state_dict(
            self, state_dict: Any, module: torch.nn.Module, keys: List[str], i: int = 0) -> None:
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
        return

    def load_networks(self, epoch: int) -> None:
        """Load all the networks from the disk.
        """
        for name, module in self.modules.items():
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if isinstance(module, torch.nn.DataParallel):
                    module = module.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, module, key.split('.'))
                module.load_state_dict(state_dict)
        return

    def print_networks(self, verbose: bool) -> None:
        """Print the total number of parameters in the network and (if verbose) network architecture
        """
        print('---------- Networks initialized -------------')
        for name, module in self.modules.items():
            num_params = 0
            for param in module.parameters():
                num_params += param.numel()
            if verbose:
                print(module)
            print('[Network %s] Total number of parameters: %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        return
