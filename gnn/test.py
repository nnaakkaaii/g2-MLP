import argparse

from src.dataloaders import dataloaders
from src.datasets import datasets
from src.loggers import loggers
from src.models import models
from src.options.test_option import TestOption
from src.transforms import transforms


def modify_test_options(opt: argparse.Namespace) -> argparse.Namespace:
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    return opt


def test(opt: argparse.Namespace):
    test_transform = transforms[opt.test_transform_name](opt)
    test_dataset = datasets[opt.dataset_name](test_transform, is_train=False, opt=opt)
    test_dataloader = dataloaders[opt.dataloader_name](test_dataset, opt)
    test_dataset_size = len(test_dataset)
    print('The number of test images = %s' %test_dataset_size)

    model = models[opt.model_name](opt)
    model.setup(opt)

    logger = loggers[opt.logger_name](model, opt)
    logger.set_dataset_length(test_dataset_size)

    model.eval_mode()

    for epoch in range(opt.num_test):
        logger.start_epoch()
        for data in test_dataloader:
            model.set_input(data)
            model.test()
            logger.end_train_iter()

    return


if __name__ == '__main__':
    opt = TestOption().parse()
    test(opt)
