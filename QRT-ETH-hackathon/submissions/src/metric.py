import numpy as np


def weighted_sign_accuracy(y_true, y_pred) -> float:
    """Competition metric: sum |y| * 1[sign(pred)==sign(y)] / sum |y|.
    A prediction of exactly 0 is always wrong."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.abs(y_true)
    correct = (np.sign(y_pred) == np.sign(y_true)) & (y_pred != 0)
    return float((w * correct).sum() / w.sum())
