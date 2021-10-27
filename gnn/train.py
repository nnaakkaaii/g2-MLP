import json
import os
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torchnet as tnt
from src.datasets import datasets
from src.models.losses import losses
from src.models.networks import networks
from src.models.optimizers import optimizers
from src.options.train_option import TrainOption
from src.transforms import transforms
from src.utils.fix_seed import fix_seed
from torch_geometric.data import Data
from torchnet.engine import Engine
from torchnet.logger.visdomlogger import VisdomPlotLogger
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
fix_seed(42)


def mlflow_setup():
    os.makedirs(opt.mlflow_root_dir, exist_ok=True)
    mlflow.set_tracking_uri(opt.mlflow_root_dir)

    if not bool(mlflow.get_experiment_by_name(opt.name)):
        mlflow.create_experiment(opt.name, artifact_location=None)

    mlflow.set_experiment(opt.name)
    mlflow.start_run(run_name=opt.run_name)
    mlflow.log_params(vars(opt))
    return


def processor(sample: Tuple[Data, bool]) -> Tuple[Any, Any]:
    data_, training = sample
    data_.to(device)

    network.train(training)

    classes = network(data_)
    loss_ = loss(classes, data_.y)
    return loss_, classes


def on_sample(state: Dict[str, Any]) -> None:
    state['sample'].y = state['sample'].y.flatten()
    if not opt.is_regression:
        state['sample'].y = state['sample'].y.long()
    state['sample'] = state['sample'], state['train']
    return


def _reset_meters() -> None:
    loss_averager.reset()
    metric_averager.reset()
    return


def on_forward(state: Dict[str, Any]):
    loss_averager.add(state['loss'].detach().cpu().item())
    metric_averager.add(state['output'].detach().cpu(), state['sample'][0].y)
    return


def on_start_epoch(state: Dict[str, Any]) -> None:
    _reset_meters()
    return


def on_end_epoch(state: Dict[str, Any]) -> None:
    current_loss = loss_averager.value()[0]
    current_metric = metric_averager.value()[0]
    train_loss_logger.log(state['epoch'], current_loss, name='fold_' + str(fold_number))
    train_metric_logger.log(state['epoch'], current_metric, name='fold_' + str(fold_number))
    fold_history['train_loss'].append(current_loss)
    fold_history['train_metric'].append(current_metric)

    _reset_meters()
    with torch.no_grad():
        engine.test(processor, test_dataset)

    current_loss = loss_averager.value()[0]
    current_metric = metric_averager.value()[0]
    test_loss_logger.log(state['epoch'], current_loss, name='fold_' + str(fold_number))
    test_metric_logger.log(state['epoch'], current_metric, name='fold_' + str(fold_number))
    fold_history['test_loss'].append(current_loss)
    fold_history['test_metric'].append(current_metric)

    metrics = {
        'train_loss': fold_history['train_loss'][-1],
        'train_metric': fold_history['train_metric'][-1],
        'test_loss': fold_history['test_loss'][-1],
        'test_metric': fold_history['test_metric'][-1],
    }
    mlflow.log_metrics(metrics, step=state['epoch'])

    # save model at every fold
    epoch = state['epoch']
    if epoch % opt.save_freq == 0:
        save_path = os.path.join(fold_save_dir, f'net_{epoch}.pth')
        os.makedirs(fold_save_dir, exist_ok=True)
        torch.save(network.state_dict(), save_path)
    return


if __name__ == '__main__':
    opt = TrainOption().parse()
    mlflow_setup()

    over_save_dir = os.path.join(opt.save_dir, opt.name)
    over_history: DefaultDict[str, List[float]] = defaultdict(list)  # 全foldの結果
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    os.makedirs(over_save_dir, exist_ok=True)
    with open(os.path.join(over_save_dir, 'options.json'), 'w') as f:
        json.dump(vars(opt), f)

    train_transform = transforms[opt.train_transform_name](opt)
    test_transform = transforms[opt.test_transform_name](opt)

    engine = Engine()
    loss_averager = tnt.meter.AverageValueMeter()
    metric_averager = tnt.meter.MSEMeter() if opt.is_regression else tnt.meter.ClassErrorMeter(accuracy=True)
    train_loss_logger = VisdomPlotLogger('line', env=opt.name, opts={'title': 'Train Loss'})
    train_metric_logger = VisdomPlotLogger('line', env=opt.name, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=opt.name, opts={'title': 'Test Loss'})
    test_metric_logger = VisdomPlotLogger('line', env=opt.name, opts={'title': 'Test Accuracy'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    train_iter = tqdm(range(1, 11), desc='Training Model......')
    for fold_number in train_iter:
        fold_save_dir = os.path.join(over_save_dir, f'{fold_number:02}')
        fold_history: DefaultDict[str, List[float]] = defaultdict(list)  # 各foldの結果

        train_dataset = datasets[opt.dataset_name](train_transform, True, fold_number, opt)

        test_dataset = datasets[opt.dataset_name](test_transform, False, fold_number, opt)

        num_features, num_classes = train_dataset.num_features, train_dataset.num_classes

        loss = losses[opt.loss_name](opt)
        network = networks[opt.network_name](num_features, num_classes, opt)

        network.to(device)

        optimizer = optimizers[opt.optimizer_name](network.parameters(), opt)

        engine.train(processor, train_dataset, maxepoch=opt.n_epochs, optimizer=optimizer)

        for key, value in fold_history.items():
            over_history[key].append(value[-1])

        train_iter.set_description(
            '[Fold %d] Training Accuracy: %.2f%% Testing Accuracy: %.2f%%' % (
                fold_number,
                fold_history['train_metric'][-1],
                fold_history['test_metric'][-1],
            )
        )

        os.makedirs(fold_save_dir, exist_ok=True)
        with open(os.path.join(fold_save_dir, 'history.json'), 'w') as f:
            json.dump(fold_history, f)

    print(
        'Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
        (
            np.array(over_history['train_metric']).mean(),
            np.array(over_history['train_metric']).std(),
            np.array(over_history['test_metric']).mean(),
            np.array(over_history['test_metric']).std(),
        )
    )

    with open(os.path.join(over_save_dir, 'history.json'), 'w') as f:
        json.dump(over_history, f)

    mlflow.end_run()
