import abc

import torch.utils.data as data


class BaseDataset(data.Dataset, metaclass=abc.ABCMeta):
    """Datasetを実装する際はこのクラスを継承する
    実装の際には、trainであればtransformを適用し、valであれば適用しないことに注意
    また、is_trainとis_valで読み込むデータセットの範囲を変更する
    """
    def __init__(self, max_dataset_size: int, dataset_length: int, is_train: bool) -> None:
        self.max_dataset_size = int(max_dataset_size)
        self.dataset_length = dataset_length
        self.is_train = is_train

    def __len__(self) -> int:
        return min(self.dataset_length, self.max_dataset_size)
