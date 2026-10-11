"""Offline CTR metrics: ranking (AUC), probability quality (LogLoss), and global calibration.

Why three: AUC only looks at the ORDER of predictions (multiply every p by 0.5 and AUC is
unchanged), LogLoss punishes wrong probabilities, and mean predicted CTR vs true CTR shows a
global bias that both can partly hide. Bidding uses the probability itself, so order alone is
not enough.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-7  # keeps log() finite when a prediction is exactly 0 or 1


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Probability that a random clicked row is scored above a random non-clicked row."""
    return float(roc_auc_score(y_true, y_prob))


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean binary cross-entropy -[y log p + (1-y) log(1-p)], the quantity training minimizes."""
    p = np.clip(y_prob.astype(np.float64), EPS, 1 - EPS)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ctr_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """All metrics in one dict. calib_ratio = mean predicted / true CTR (1.0 = no global bias)."""
    mean_pred, true_ctr = float(np.mean(y_prob)), float(np.mean(y_true))
    return {
        "auc": auc(y_true, y_prob),
        "logloss": logloss(y_true, y_prob),
        "mean_pred": mean_pred,
        "true_ctr": true_ctr,
        "calib_ratio": mean_pred / true_ctr,
    }
