import argparse
from typing import Any

from torchvision.transforms import transforms


def create_transform(opt: argparse.Namespace) -> Any:
    return create_affine_transform(
        opt.transform_degree, opt.transform_translate_x, opt.transform_translate_y, opt.transform_scale_min, opt.transform_scale_max)


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--transform_degree', type=int, default=15, help='augmentationとして傾ける角度')
    parser.add_argument('--transform_translate_x', type=float, default=0.1, help='x方向の移動割合')
    parser.add_argument('--transform_translate_y', type=float, default=0.1, help='y方向の移動割合')
    parser.add_argument('--transform_scale_min', type=float, default=0.8, help='縮小の最小割合')
    parser.add_argument('--transform_scale_max', type=float, default=1.2, help='拡大の最大割合')
    return parser


def create_affine_transform(degree: int, translate_x: float, translate_y: float, scale_min: float, scale_max: float):
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=[-degree, degree],
            translate=(translate_x, translate_y),
            scale=(scale_min, scale_max),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
