import argparse

from src.datasets import datasets
from src.transforms import transforms
from src.dataloaders import dataloaders
from src.models import models
from src.loggers import loggers
from src.options.train_option import TrainOption
from src.utils.fix_seed import fix_seed


fix_seed(42)


def train(opt: argparse.Namespace) -> None:
    train_transform = transforms[opt.train_transform_name](opt)
    train_dataset = datasets[opt.dataset_name](train_transform, True, opt)
    train_dataloader = dataloaders[opt.dataloader_name](train_dataset, opt)
    train_dataset_size = len(train_dataset)
    train_dataloader_size = len(train_dataloader)
    print('The number of training images = %s' % train_dataset_size)

    val_transform = transforms[opt.val_transform_name](opt)
    val_dataset = datasets[opt.dataset_name](val_transform, True, opt)
    val_dataloader = dataloaders[opt.dataloader_name](val_dataset, opt)
    val_dataset_size = len(val_dataset)
    val_dataloader_size = len(val_dataloader)
    print('The number of validating images = %s' % val_dataset_size)

    model = models[opt.model_name](opt)
    model.setup(opt)

    logger = loggers[opt.logger_name](model, opt)
    logger.set_dataset_length(train_dataloader_size, val_dataloader_size)
    logger.save_options()

    for epoch in range(opt.epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        logger.start_epoch()
        model.train_mode()
        for data in train_dataloader:
            model.set_input(data)
            model.optimize_parameters()
            logger.end_train_iter()
        for data in val_dataloader:
            model.set_input(data)
            model.test()
            logger.end_val_iter()
        logger.end_epoch()
        model.update_learning_rate()
    logger.end_all_training()
    return


if __name__ == '__main__':
    opt = TrainOption().parse()
    train(opt)
