import argparse
from typing import Any

from torchvision.transforms import transforms


def create_transform(opt: argparse.Namespace) -> Any:
    return create_no_transform()


def create_no_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
