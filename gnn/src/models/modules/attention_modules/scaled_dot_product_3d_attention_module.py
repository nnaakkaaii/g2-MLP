import argparse

import torch
import torch.nn as nn

from . import _attention_module


def attention_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = _attention_module.attention_module_modify_commandline_options(parser)
    return parser


def create_attention_module(attention_hidden_dim: int, opt: argparse.Namespace) -> nn.Module:
    return _attention_module.AttentionModule(ScaledDotProduct3DAttentionProduct(), opt.attention_dropout_rate)


class ScaledDotProduct3DAttentionProduct(_attention_module.BaseAttentionProduct):
    """
    Scaled Dot Product Attentionの実装
        Dropout(Softmax(Q, transposed(K))) V
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        logit = bmm(query, transposed(key)) / sqrt(h)

        :param query: クエリ (n, a, h)
        :param key: キー (n, b, h)
        :return: (n, a, b)
        """
        hidden_dim = query.size(-1)  # h
        # transposed_key: (n, h, b)
        transposed_key = torch.transpose(key, 1, 2)  # batchを固定して転置
        # logit: (n, a, b)
        logit = torch.bmm(query, transposed_key)
        logit /= hidden_dim**-0.5  # scaled
        return logit
