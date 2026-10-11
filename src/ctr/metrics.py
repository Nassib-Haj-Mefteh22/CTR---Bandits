import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob.astype(np.float64), EPS, 1 - EPS)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ctr_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    mean_pred, true_ctr = float(np.mean(y_prob)), float(np.mean(y_true))
    return {
        "auc": auc(y_true, y_prob),
        "logloss": logloss(y_true, y_prob),
        "mean_pred": mean_pred,
        "true_ctr": true_ctr,
        "calib_ratio": mean_pred / true_ctr,
    }
