import abc

import torch
import torch.nn as nn


class BaseMarginProduct(nn.Module, metaclass=abc.ABCMeta):
    """is_train=Falseであれば、ペナルティを与えない
    """
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter()

    def initialize_weight(self, input_dim: int, output_dim: int) -> None:
        self.weight = nn.Parameter(torch.FloatTensor(output_dim, input_dim))

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, is_train: bool) -> torch.Tensor:
        pass


class XFace(nn.Module):
    """metric_fc層を結合 (softmaxはlossに含まれる)
    """
    def __init__(self, xface_product_class: BaseMarginProduct, out_dim: int, last_p_dim: int) -> None:
        super().__init__()

        xface_product_class.initialize_weight(last_p_dim, out_dim)
        self.label = torch.Tensor()  # 先にインスタンス変数labelを設定しておき、set_inputの度に外部から変更する
        self.is_train = True  # 先にインスタンス変数is_trainを設定しておき、train_modeやeval_modeの度などに外部から変更する
        self.model = xface_product_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.label, self.is_train)
