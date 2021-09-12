import argparse

from src.datasets import datasets
from src.transforms import transforms
from src.dataloaders import dataloaders
from src.models import models
from src.options.test_option import TestOption


def modify_test_options(opt: argparse.Namespace) -> argparse.Namespace:
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    return opt


if __name__ == '__main__':
    opt = TestOption().parse()
    opt = modify_test_options(opt)
    transform = transforms[opt.transform_name](opt)
    dataset = datasets[opt.dataset_name](transform, opt)
    dataloader = dataloaders[opt.dataloader_name](dataset, opt)
    model = models[opt.model_name](opt)
    model.setup(opt)

    model.eval()
    for data in dataloader:
        model.set_input(data)
        model.test()
        # TODO: 未作成
