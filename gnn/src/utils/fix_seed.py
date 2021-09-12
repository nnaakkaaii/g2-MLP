import random
import numpy as np
import torch


def fix_seed(seed: int = 42) -> None:
    """seed値を固定させることで機械学習の再現性を保たせる
    [参考]
    https://qiita.com/si1242/items/d2f9195c08826d87d6ad
    Args:
        seed (int): seedの設定値. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return
