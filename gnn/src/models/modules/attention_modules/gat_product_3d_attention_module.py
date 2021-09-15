import argparse

import torch
import torch.nn as nn

from . import _attention_module


def attention_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = _attention_module.attention_module_modify_commandline_options(parser)
    parser.add_argument('--leakyrelu_alpha', type=float, default=0.2, help='leakyreluのalpha')
    return parser


def create_attention_module(attention_hidden_dim: int, opt: argparse.Namespace) -> nn.Module:
    return _attention_module.AttentionModule(GATProduct3DAttentionProduct(attention_hidden_dim, opt.leakyrelu_alpha), opt.attention_dropout_rate)


class GATProduct3DAttentionProduct(_attention_module.BaseAttentionProduct):
    """
    GAT Product Attentionの実装
    """
    def __init__(self, attention_hidden_dim: int, leakyrelu_alpha: float) -> None:  # TODO : 引数 concat
        """
        :param attention_hidden_dim: attentionの出力次元
        :param leakyrelu_alpha: leakyreluのハイパーパラメータ
        """
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim
        self.vector_a = nn.Parameter(torch.empty(size=(2*attention_hidden_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(leakyrelu_alpha)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        logit = cat(repeated(query), repeated(key)) a

        :param query: クエリ (n, a, h)  # a : embedding_dim (ノード数), h : hid_dim (attentionの出力次元)
        :param key: キー (n, a, h)
        :return: (n, a, a)
        """
        qa = torch.matmul(query, self.vector_a[:self.attention_hidden_dim])  # (n, a, h) x (h, 1) -> (n, a, 1)
        ka = torch.matmul(key, self.vector_a[self.attention_hidden_dim:])  # (n, a, h) x (h, 1) -> (n, a, 1)
        # broadcast add
        e = qa + ka.transpose(2, 1)  # (n, a, 1) + (n, 1, a) -> (n, a, a) + (n, a, a)  (: repeated) -> (n, a, a)
        return self.leakyrelu(e)
