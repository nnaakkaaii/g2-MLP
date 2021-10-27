import argparse

import torch.nn as nn


def create_loss(opt: argparse.Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()
