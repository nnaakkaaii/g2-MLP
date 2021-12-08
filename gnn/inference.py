import json
import os
import csv

import matplotlib.pyplot as plt
import torch
from src.datasets import datasets
from src.models.networks import networks
from src.options.train_option import BaseOption
from src.transforms import transforms
from src.transforms.label_normalize_transform import create_transform
from src.utils.engine import Engine


class Logger:
    def __init__(self, opt, device, result_dir=None, denormalize=None):
        self.device = device
        self.regression = opt.regression
        self.name = opt.name
        self.result_dir = result_dir if result_dir is not None else opt.save_dir
        self.denormalize = denormalize

    def on_sample(self, state):
        if isinstance(state['input'], list):
            # for Multi-GPU
            state['input'] = [data.to(self.device) for data in state['input']]
            state['label'] = torch.cat([data.y for data in state['input']], dim=0).to(self.device)
        else:
            state['input'] = state['input'].to(self.device)
            state['label'] = state['input'].y
        return

    def on_forward(self, state):
        iteration = state['iteration']

        data = state['input'].clone().cpu().detach()

        if not self.regression:
            fig = plt.figure()
            for phase in ('label', 'pred'):
                title = f'{self.name}_iter{iteration}_{phase}.png'
                if phase == 'label':
                    y = state['label']
                else:
                    y = torch.argmax(state['output'], dim=1)
                y = y.clone().cpu().detach()
                sc = plt.scatter(data.pos[:, 0], data.pos[:, 1], c=y)
                plt.colorbar(sc)
                plt.title(title)
                plt.xlabel('x axis')
                plt.ylabel('y axis')
                plt.grid()
                fig.savefig(os.path.join(self.result_dir, title))
                plt.close()
        else:
            for phase in ('label', 'pred'):
                title = f'{self.name}_iter{iteration}_{phase}.csv'
                if phase == 'label':
                    y = state['label']
                else:
                    y = state['output']
                y = self.denormalize(y)
                y = y.clone().cpu().detach().numpy().tolist()
                with open(os.path.join(self.result_dir, title), 'w') as f:
                    csv.writer(f).writerows(y)


    def on_start_test(self, state):
        pass

    def on_start_test_iteration(self, state):
        pass

    def on_end_test_iteration(self, state):
        pass

    def on_end_test(self, state):
        pass

    @property
    def hooks(self):
        return {
            'on_sample': self.on_sample,
            'on_forward': self.on_forward,
            'on_start_test': self.on_start_test,
            'on_start_test_iteration': self.on_start_test_iteration,
            'on_end_test_iteration': self.on_end_test_iteration,
            'on_end_test': self.on_end_test,
        }


def inference(opt):
    save_dir = os.path.join(opt.save_dir, opt.name)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    with open(os.path.join(save_dir, 'options.json')) as f:
        for k, v in json.load(f).items():
            setattr(opt, k, v)

    denormalize = None
    if hasattr(opt, 'mean') and hasattr(opt, 'std'):
        denormalize = create_transform(opt).denormalize
    logger = Logger(opt, device, result_dir=save_dir, denormalize=denormalize)

    engine = Engine()
    engine.hooks.update(logger.hooks)

    test_transform = transforms[opt.val_transform_name](opt)
    test_dataset = datasets[opt.dataset_name](test_transform, False, opt)

    num_features, num_classes = test_dataset.num_features, test_dataset.num_classes
    network = networks[opt.network_name](num_features, num_classes, opt)
    network.to(device)

    net_path = os.path.join(save_dir, 'net_last.pth')
    network.load_state_dict(torch.load(net_path))

    engine.inference(network, test_dataset)


if __name__ == '__main__':
    opt = BaseOption().parse()
    inference(opt)
