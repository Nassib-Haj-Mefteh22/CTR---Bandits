import numpy as np
from sklearn.metrics import roc_auc_score

from ctr.calibrate import ece, fit_platt, fit_temperature, reliability_bins, sigmoid


def _calibrated_sample(n: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.0, 1.5, n)
    y = (rng.random(n) < sigmoid(z)).astype(np.float64)
    return z, y


def test_ece_near_zero_when_labels_drawn_from_predictions():
    z, y = _calibrated_sample()
    assert ece(y, sigmoid(z)) < 0.005


def test_ece_of_constant_prediction_is_the_gap_to_base_rate():
    y = np.array([1, 0, 0, 0], dtype=np.float64)
    assert abs(ece(y, np.full(4, 0.6)) - 0.35) < 1e-12


def test_reliability_bins_counts_and_edge_value():
    y = np.array([0, 1, 1], dtype=np.float64)
    p = np.array([0.0, 0.5, 1.0])
    conf, acc, count = reliability_bins(y, p, n_bins=15)
    assert count.sum() == 3 and count[0] == 1 and count[7] == 1 and count[14] == 1
    assert acc[14] == 1.0 and conf[14] == 1.0


def test_temperature_recovers_known_overconfidence():
    z, y = _calibrated_sample()
    t = fit_temperature(2.0 * z, y)
    assert abs(t - 2.0) < 0.05
    assert ece(y, sigmoid(2.0 * z / t)) < ece(y, sigmoid(2.0 * z))


def test_calibration_maps_leave_auc_unchanged():
    z, y = _calibrated_sample(20_000)
    t = fit_temperature(z, y)
    a, b = fit_platt(z + 0.7, y)
    assert abs(b + 0.7 * a) < 0.1
    auc = roc_auc_score(y, z)
    assert roc_auc_score(y, sigmoid(z / t)) == auc
    assert roc_auc_score(y, sigmoid(a * (z + 0.7) + b)) == auc
