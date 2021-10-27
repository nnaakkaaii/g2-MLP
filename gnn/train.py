import argparse

import torch

from src.dataloaders import dataloaders
from src.datasets import datasets
from src.loggers import loggers
from src.models.losses import losses
from src.models.networks import networks
from src.models.optimizers import optimizers
from src.options.train_option import TrainOption
from src.transforms import transforms
from src.utils.fix_seed import fix_seed
from torchnet.engine import Engine

fix_seed(42)


def train(opt: argparse.Namespace) -> None:
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    train_transform = transforms[opt.train_transform_name](opt)
    test_transform = transforms[opt.test_transform_name](opt)

    engine = Engine()
    logger = loggers[opt.logger_name](opt)

    for i in range(1, 11):
        print(f'************************* FOLD {i:02} STARTED *************************')
        train_dataset = datasets[opt.dataset_name](train_transform, True, i, opt)
        train_dataloader = dataloaders[opt.dataloader_name](train_dataset, True, opt)

        test_dataset = datasets[opt.dataset_name](test_transform, False, i, opt)
        test_dataloader = dataloaders[opt.dataloader_name](test_dataset, False, opt)

        num_features, num_classes = train_dataset.num_features, train_dataset.num_classes

        loss = losses[opt.loss_name](opt)
        network = networks[opt.network_name](num_features, num_classes, opt)
        network.to(device)

        def processor(sample):
            data_, training = sample
            data_.to(device)

            network.train(training)

            classes = network(data_)
            loss_ = loss(classes, data_.y)
            return loss_, classes

        logger.set_network(network)
        if i == 0 and opt.verbose:
            logger.print_networks()

        logger.set_test_callback(lambda: engine.test(processor, test_dataloader))
        engine.hooks.update(logger.hooks)

        optimizer = optimizers[opt.optimizer_name](network.parameters(), opt)

        engine.train(processor, train_dataloader, maxepoch=opt.n_epochs, optimizer=optimizer)

    logger.on_end_all_training()
    return


if __name__ == '__main__':
    opt = TrainOption().parse()
    train(opt)
