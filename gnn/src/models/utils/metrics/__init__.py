from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='micro'),
        'recall': recall_score(y_true, y_pred, average='micro'),
        'f1': f1_score(y_true, y_pred, average='micro'),
    }
