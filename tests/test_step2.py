import numpy as np
import pytest
import torch
from torch import nn

from ctr.data import CTRDataset, make_loader
from ctr.metrics import auc, ctr_metrics, logloss
from ctr.models.logreg import LogisticRegression
from ctr.train import TrainConfig, fit, set_seed


def _random_batch(n: int, n_ids: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(n, 26, generator=g)
    sparse = torch.randint(0, n_ids, (n, 26), generator=g)
    label = torch.randint(0, 2, (n,), generator=g).float()
    return dense, sparse, label


# --- 2.1 model ---------------------------------------------------------------------------------
def test_logreg_forward_shape_and_initial_logit_is_bias_plus_dense():
    model = LogisticRegression(n_ids=100)
    dense, sparse, _ = _random_batch(8, 100)
    out = model(dense, sparse)
    assert out.shape == (8,)
    # sparse weights start at zero, so only the dense linear part contributes
    torch.testing.assert_close(out, model.dense(dense).squeeze(1))


def test_logreg_equals_one_hot_linear_model():
    """Embedding-sum must equal w . one_hot(x): the definition of logistic regression."""
    torch.manual_seed(0)
    model = LogisticRegression(n_ids=50)
    nn.init.normal_(model.sparse_weight.weight)
    dense, sparse, _ = _random_batch(4, 50)
    one_hot = torch.zeros(4, 50).scatter_add_(1, sparse, torch.ones_like(sparse, dtype=torch.float))
    expected = one_hot @ model.sparse_weight.weight.squeeze(1) + model.dense(dense).squeeze(1)
    torch.testing.assert_close(model(dense, sparse), expected)


# --- 2.3 metrics -------------------------------------------------------------------------------
def test_auc_perfect_and_constant():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.full(4, 0.3)) == 0.5
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_logloss_known_values():
    y = np.array([1, 0])
    assert logloss(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert logloss(y, np.array([1.0, 0.0])) < 1e-6  # clipped, finite
    # predicting the base rate gives the entropy of the labels
    y = np.array([1, 0, 0, 0])
    p = 0.25
    assert logloss(y, np.full(4, p)) == pytest.approx(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def test_ctr_metrics_calibration_ratio():
    m = ctr_metrics(np.array([1, 0, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert m["true_ctr"] == 0.25 and m["mean_pred"] == 0.5 and m["calib_ratio"] == 2.0


# --- 2.2 training loop -------------------------------------------------------------------------
def test_overfit_one_batch_reaches_near_zero_loss():
    """Sanity check of model + loss + optimizer: a model that can memorize must memorize."""
    set_seed(0)
    model = LogisticRegression(n_ids=10_000)
    dense, sparse, label = _random_batch(64, 10_000)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(dense, sparse), label)
        loss.backward()
        opt.step()
    assert loss.item() < 0.01


def test_fit_loss_decreases_and_restores_best(tmp_path):
    set_seed(0)
    n, n_ids = 2000, 500
    rng = np.random.default_rng(0)
    sparse = rng.integers(0, n_ids, (n, 26))
    true_w = rng.normal(size=n_ids)
    logits = true_w[sparse].sum(1) / 3
    label = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    ds = CTRDataset(np.zeros((n, 26), np.float32), sparse, label)
    model = LogisticRegression(n_ids=n_ids)
    cfg = TrainConfig(epochs=5, batch_size=256, lr=0.05, patience=5)
    result = fit(
        model, make_loader(ds, 256, True), make_loader(ds, 1000, False), cfg, tmp_path / "m.pt"
    )
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < losses[0]
    best = torch.load(tmp_path / "m.pt", weights_only=True)
    torch.testing.assert_close(model.state_dict()["sparse_weight.weight"],
                               best["sparse_weight.weight"])
