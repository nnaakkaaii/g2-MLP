import argparse

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
from tqdm import tqdm

fix_seed(42)


def train(opt: argparse.Namespace) -> None:
    device = opt.device

    train_transform = transforms[opt.train_transform_name](opt)

    test_transform = transforms[opt.test_transform_name](opt)

    engine = Engine()
    logger = loggers[opt.logger_name](opt)

    train_iter = tqdm(range(1, 11), desc='Training Model......')
    for i in train_iter:
        network = networks[opt.network_name](opt)
        network.to(device)

        loss = losses[opt.loss_name](opt)

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

        train_dataset = datasets[opt.dataset_name](train_transform, True, i, opt)
        train_dataloader = dataloaders[opt.dataloader_name](train_dataset, True, opt)

        test_dataset = datasets[opt.dataset_name](test_transform, False, i, opt)
        test_dataloader = dataloaders[opt.dataloader_name](test_dataset, False, opt)

        logger.set_test_callback(lambda: engine.test(processor, test_dataloader))
        engine.hooks.update(logger.hooks)

        optimizer = optimizers[opt.optimizer_name](network.parameters(), opt)

        engine.train(network, train_dataloader, maxepoch=opt.n_epochs, optimizer=optimizer)

    logger.on_end_all_training()
    return


if __name__ == '__main__':
    opt = TrainOption().parse()
    train(opt)
