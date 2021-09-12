from typing import Optional

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Scaled Dot Attentionの実装
        Dropout(Softmax(Q, transposed(K))) V
    """
    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        attention計算
            - logit = bmm(query, transposed(key))
            - attention_weight = softmax(logit)
            - context_vector = bmm(attention_weight, value)
        ただし、maskされた領域のlogitは-infに

        :param query: クエリ (n, a, h)
        :param key: キー (n, b, h)
        :param value: バリュー (n, b, h)
        :param mask: マスク (n, b)
        :return: (n, a, h)
        """
        hidden_dim = query.size(-1)  # h
        # transposed_key: (n, h, b)
        transposed_key = torch.transpose(key, 1, 2)  # batchを固定して転置
        # logit: (n, a, b)
        logit = torch.bmm(query, transposed_key)
        logit /= hidden_dim**-0.5  # scaled
        if mask is not None:
            if mask.dim() == 2:  # encoderのmaskやsource target attentionのmaskでpaddingをmaskするため
                logit.masked_fill_(mask.unsqueeze(1), -float('inf'))  # mask = Trueの領域を -inf に
            elif mask.dim() == 3:  # decoderのmaskで先読みをなくすため
                logit.masked_fill_(mask, -float('inf'))
        # attention_weight: (n, a, b)
        attention_weight = self.softmax(logit)
        attention_weight = self.dropout(attention_weight)
        # context_vector: (n, a, h)
        context_vector = torch.bmm(attention_weight, value)
        return context_vector
