import json
import os
from copy import deepcopy

import numpy as np
import optuna
from src.options.train_option import TrainOption
from src.utils.fix_seed import fix_seed
from train import train


def objective(trial):
    conf = deepcopy(opt)
    trial_id = str(trial._trial_id)
    conf.save_dir = os.path.join(conf.save_dir, conf.name, trial_id)
    conf.run_name = conf.save_dir + trial_id

    conf.n_epochs = trial.suggest_categorical('n_epochs', [50, 100, 200, 300, 400, 500])

    # network hyper parameters
    if conf.network_name == 'gmlp_node_classification':
        conf.hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])  # afbd8d67521466b227665efc0c7078ba339e4341
        conf.ffn_dim = trial.suggest_categorical('ffn_dim', [64, 128, 256, 512, 1024])  # afbd8d67521466b227665efc0c7078ba339e4341
        conf.n_layers = trial.suggest_int('n_layers', 2, 8)  # afbd8d67521466b227665efc0c7078ba339e4341
        # conf.prob_survival = trial.suggest_categorical('prob_survival', [0.6, 0.8, 1.0])  # afbd8d67521466b227665efc0c7078ba339e4341
    elif conf.network_name == 'gmlp_graph_classification':
        conf.hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
        conf.ffn_dim = trial.suggest_categorical('ffn_dim', [64, 128, 256, 512, 1024])
        conf.n_layers = trial.suggest_int('n_layers', 2, 8)
        # conf.prob_survival = trial.suggest_categorical('prob_survival', [0.6, 0.8, 1.0])  # afbd8d67521466b227665efc0c7078ba339e4341
    elif conf.network_name == 'gmlp_hierarchical_sagpool_graph_classification':
        conf.hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
        conf.ffn_dim = trial.suggest_categorical('ffn_dim', [64, 128, 256, 512, 1024])
        conf.n_layers = trial.suggest_int('n_layers', 2, 5)
        conf.pool_ratio = trial.suggest_uniform('pool_ratio', 0.1, 0.5)
        conf.n_hierarchies = trial.suggest_int('n_hierarchies', 2, 6)
        conf.version = trial.suggest_categorical('version', [1, 2, 3, 4, 5, 6, 7])
    else:
        raise NotImplementedError

    # optimizer hyper parameters
    if conf.optimizer_name == 'adam':
        conf.lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)  # afbd8d67521466b227665efc0c7078ba339e4341
        # conf.beta1 = trial.suggest_uniform('beta1', 0.0, 1.0)  # afbd8d67521466b227665efc0c7078ba339e4341
        # conf.beta2 = trial.suggest_uniform('beta2', 0.0, 1.0)  # afbd8d67521466b227665efc0c7078ba339e4341
    else:
        raise NotImplementedError

    # scheduler hyper parameters
    if conf.scheduler_name == 'step':
        conf.lr_decay_iters = trial.suggest_categorical('lr_decay_iters', [10, 20, 30, 40, 50])  # afbd8d67521466b227665efc0c7078ba339e4341
        conf.lr_decay_gamma = trial.suggest_uniform('lr_decay_gamma', 0.0, 1.0)  # afbd8d67521466b227665efc0c7078ba339e4341
    else:
        raise NotImplementedError

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    for i in range(1, 6):
        fix_seed(i)

        current_history = train(opt)

        # update history
        for key, value in current_history.items():
            history[key].append(value)

    return np.mean(history['val_accuracy'])


if __name__ == '__main__':
    opt = TrainOption().parse()
    study = optuna.create_study(study_name=opt.name, storage='sqlite:///./sqlite3.db', load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=300)

    with open(os.path.join(opt.save_dir, 'optuna_history.json'), 'w') as f:
        optuna_history = {
            'optuna_history': [
                {
                    'id': trial._trial_id,
                    'value': trial.value,
                    'datetime_start': str(trial.datetime_start),
                    'datetime_complete': str(trial.datetime_complete),
                    'params': trial.params,
                } for trial in study.trials
            ],
        }
        json.dump(optuna_history, f)

    with open(os.path.join(opt.save_dir, 'optuna_best_history.json'), 'w') as f:
        optuna_best_history = {
            'optuna_best_history': {
                'id': study.best_trial._trial_id,
                'value': study.best_trial.value,
                'datetime_start': study.best_trial.datetime_start,
                'datetime_complete': study.best_trial.datetime_complete,
                'params': study.best_trial.params,
            }
        }
        json.dump(optuna_best_history, f)
