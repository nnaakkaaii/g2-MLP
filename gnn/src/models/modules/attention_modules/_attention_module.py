import abc
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f


def attention_module_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--attention_dropout_rate', type=float, default=0.1, help='attentionのsoftmaxのdropout_rate')
    return parser


class BaseAttentionProduct(nn.Module, metaclass=abc.ABCMeta):
    """AttentionProduct(Q, K)
    """
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        pass


class AttentionModule(nn.Module):
    """Dropout(Softmax(AttentionProduct(Q, K)))V
    """
    def __init__(self, attention_product: BaseAttentionProduct, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.model = attention_product
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_rate = dropout_rate

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attention計算
            - logit = attention_product(query, key)
            - attention_weight = softmax(logit)
            - context_vector = bmm(attention_weight, value)
        ただし、maskされた領域のlogitは-infに

        :param query: クエリ (n, a, h)
        :param key: キー (n, b, h)
        :param value: バリュー (n, b, h)
        :param mask: マスク (n, b)
        :return: (n, a, h)
        """
        # logit: (n, a, b)
        logit = self.model(query, key)
        if mask is not None:
            if mask.dim() == 2:  # encoderのmaskやsource target attentionのmaskでpaddingをmaskするため
                logit.masked_fill_(mask.unsqueeze(1), -float('inf'))  # mask = Trueの領域を -inf に
            elif mask.dim() == 3:  # decoderのmaskで先読みをなくすため
                logit.masked_fill_(mask, -float('inf'))
        # attention_weight: (n, a, b)
        attention_weight = self.softmax(logit)
        attention_weight = f.dropout(attention_weight, p=self.dropout_rate, training=self.training)
        # context_vector: (n, a, h)
        context_vector = torch.bmm(attention_weight, value)
        return context_vector
