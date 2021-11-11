import json
import os

import mlflow
import numpy as np
import torch
import torchnet as tnt
from tqdm import tqdm
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

from src.datasets import datasets
from src.models.losses import losses
from src.models.networks import networks
from src.models.optimizers import optimizers
from src.models.schedulers import schedulers
from src.options.train_option import TrainOption
from src.transforms import transforms
from src.utils.fix_seed import fix_seed
from src.utils.engine import Engine

fix_seed(42)


class Logger:
    def __init__(self, opt, device, result_dir=None):
        self.current_history = {
            'train_loss': None,
            'train_accuracy': None,
            'val_loss': None,
            'val_accuracy': None,
        }
        # 以下では静的データのみインスタンス変数化する. 動的データはstateで保管する.
        self.opt = opt  # 保存用
        self.device = device
        self.result_dir = result_dir if result_dir is not None else opt.save_dir
        self.name = opt.name
        self.save_freq = opt.save_freq
        os.makedirs(self.result_dir, exist_ok=True)
        # mlflow configurations
        self.run_name = opt.run_name
        self.mlflow_root_dir = opt.mlflow_root_dir
        os.makedirs(self.mlflow_root_dir, exist_ok=True)

    def on_sample(self, state):
        if isinstance(state['input'], list):
            state['input'] = [data.to(self.device) for data in state['input']]
            state['label'] = torch.cat([data.y for data in state['input']], dim=0).to(self.device)
        else:
            state['input'] = state['input'].to(self.device)
            state['label'] = state['input'].y
        return

    def on_forward(self, state):
        state['loss_averager'].add(state['loss'].detach().cpu().item())
        if state['output'].dim() == state['label'].dim() == 2:
            # MCEと同じ形式に揃える
            state['output'] = state['output'].view(-1)
            state['output'] = torch.stack([(state['output'] < 0).long(), (state['output'] > 0).long()], dim=1)
            state['label'] = state['label'].view(-1)
        state['accuracy_averager'].add(state['output'].detach().cpu(), state['label'])
        return

    def on_update(self, state):
        return

    def on_start_training(self, state):
        # setup mlflow
        mlflow.set_tracking_uri(self.mlflow_root_dir)
        if not bool(mlflow.get_experiment_by_name(self.name)):
            mlflow.create_experiment(self.name, artifact_location=None)
        mlflow.set_experiment(self.name)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(vars(self.opt))
        # setup logger
        state['loss_averager'] = tnt.meter.AverageValueMeter()
        state['accuracy_averager'] = tnt.meter.ClassErrorMeter(accuracy=True)
        # 各epoch終了ごとの結果を保存する
        state['history'] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        return

    def on_start_epoch(self, state):
        return

    def on_start_train_epoch(self, state):
        state['loss_averager'].reset()
        state['accuracy_averager'].reset()
        return

    def on_start_train_iteration(self, state):
        return

    def on_end_train_iteration(self, state):
        return

    def on_end_train_epoch(self, state):
        current_loss = state['loss_averager'].value()[0]
        current_accuracy = state['accuracy_averager'].value()[0]
        state['history']['train_loss'].append(current_loss)
        state['history']['train_accuracy'].append(current_accuracy)
        return

    def on_start_val_epoch(self, state):
        state['loss_averager'].reset()
        state['accuracy_averager'].reset()
        return

    def on_start_val_iteration(self, state):
        return

    def on_end_val_iteration(self, state):
        return

    def on_end_val_epoch(self, state):
        current_loss = state['loss_averager'].value()[0]
        current_accuracy = state['accuracy_averager'].value()[0]
        state['history']['val_loss'].append(current_loss)
        state['history']['val_accuracy'].append(current_accuracy)
        return

    def on_end_epoch(self, state):
        epoch = state['epoch']

        # save loss & metric
        metrics = {
            'train_loss': state['history']['train_loss'][-1],
            'train_accuracy': state['history']['train_accuracy'][-1],
            'val_loss': state['history']['val_loss'][-1],
            'val_accuracy': state['history']['val_accuracy'][-1],
        }
        mlflow.log_metrics(metrics, step=epoch)
        self.current_history = metrics

        # save network
        if epoch % self.save_freq == 0:
            save_path = os.path.join(self.result_dir, f'net_{epoch}.pth')
            if isinstance(state['network'], DataParallel):
                torch.save(state['network'].module.state_dict(), save_path)
            else:
                torch.save(state['network'].state_dict(), save_path)
        return

    def on_end_training(self, state):
        mlflow.end_run()
        # save loss & metric
        with open(os.path.join(self.result_dir, 'history.json'), 'w') as f:
            json.dump(state['history'], f)
        save_path = os.path.join(self.result_dir, f'net_last.pth')
        if isinstance(state['network'], DataParallel):
            torch.save(state['network'].module.state_dict(), save_path)
        else:
            torch.save(state['network'].state_dict(), save_path)
        return

    @property
    def hooks(self):
        return {
            'on_sample': self.on_sample,
            'on_forward': self.on_forward,
            'on_update': self.on_update,
            'on_start_training': self.on_start_training,
            'on_start_epoch': self.on_start_epoch,
            'on_start_train_epoch': self.on_start_train_epoch,
            'on_start_train_iteration': self.on_start_train_iteration,
            'on_end_train_iteration': self.on_end_train_iteration,
            'on_end_train_epoch': self.on_end_train_epoch,
            'on_start_val_epoch': self.on_start_val_epoch,
            'on_start_val_iteration': self.on_start_val_iteration,
            'on_end_val_iteration': self.on_end_val_iteration,
            'on_end_val_epoch': self.on_end_val_epoch,
            'on_end_epoch': self.on_end_epoch,
            'on_end_training': self.on_end_training,
        }


def train(opt, fold_number=None):
    save_dir = os.path.join(opt.save_dir, opt.name)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    if fold_number is not None:
        logger = Logger(opt, device, result_dir=os.path.join(save_dir, f'{fold_number:02}'))
    else:
        logger = Logger(opt, device, result_dir=save_dir)

    engine = Engine()
    engine.hooks.update(logger.hooks)

    # transform
    train_transform = transforms[opt.train_transform_name](opt)
    val_transform = transforms[opt.val_transform_name](opt)
    # dataloader
    if fold_number is not None:
        train_dataset = datasets[opt.dataset_name](train_transform, True, fold_number, opt)
        val_dataset = datasets[opt.dataset_name](val_transform, False, fold_number, opt)
    else:
        train_dataset = datasets[opt.dataset_name](train_transform, True, 1, opt)
        val_dataset = datasets[opt.dataset_name](val_transform, False, 1, opt)

    if len(opt.gpu_ids) > 1:
        train_dataloader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_dataloader = DataListLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    # loss
    loss = losses[opt.loss_name](opt)
    loss.to(device)
    # network
    num_features, num_classes, node_level = train_dataset.num_features, train_dataset.num_classes, train_dataset.node_level
    network = networks[opt.network_name](num_features, num_classes, node_level, opt)
    network.to(device)
    # optimizer
    if len(opt.gpu_ids) > 1:
        network = DataParallel(network, device_ids=opt.gpu_ids)
        optimizer = optimizers[opt.optimizer_name](network.module.parameters(), opt)
    else:
        optimizer = optimizers[opt.optimizer_name](network.parameters(), opt)
    # scheduler
    scheduler = schedulers[opt.scheduler_name](optimizer, opt)

    # train
    engine.train(network, train_dataloader, val_dataloader, opt.n_epochs, optimizer, scheduler, loss)

    return logger.current_history


def run(opt):
    save_dir = os.path.join(opt.save_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'options.json'), 'w') as f:
        json.dump(vars(opt), f)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    train_iter = tqdm(range(1, 6), desc='Training Model......')
    for fold_number in train_iter:
        current_history = train(opt, fold_number)

        # update history
        for key, value in current_history.items():
            history[key].append(value)

        train_iter.set_description('[Fold %d] Training Accuracy: %.2f%% Testing Accuracy: %.2f%%' % (
            fold_number, history['train_accuracy'][-1], history['val_accuracy'][-1]
        ))

    print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' % (
        np.array(history['train_accuracy']).mean(), np.array(history['train_accuracy']).std(),
        np.array(history['val_accuracy']).mean(), np.array(history['val_accuracy']).std()
    ))

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    return history


if __name__ == '__main__':
    opt = TrainOption().parse()
    run(opt)
