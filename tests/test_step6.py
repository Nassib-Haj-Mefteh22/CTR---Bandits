import numpy as np
import pytest
import torch

from bandit.policies import EpsilonGreedy, LinUCB, RandomPolicy
from bandit.simulator import BanditEnv, WhitenedPCA, arm_probs, embedding_features, top_values
from ctr.models.dcnv2 import DCNv2
from ctr.models.deepfm import DeepFM


def _toy_env(seed: int = 0, t: int = 500, k: int = 4) -> BanditEnv:
    rng = np.random.default_rng(123)
    return BanditEnv.from_probs(rng.normal(size=(t, 3)), rng.random((t, k)), seed)


def test_env_is_reproducible_for_a_seed_and_differs_across_seeds():
    a, b, c = _toy_env(seed=1), _toy_env(seed=1), _toy_env(seed=2)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    assert not np.array_equal(a.rewards, c.rewards)


def test_rewards_are_bernoulli_with_the_true_probabilities():
    probs = np.tile([0.1, 0.5, 0.9], (40_000, 1))
    env = BanditEnv.from_probs(np.zeros((40_000, 1)), probs, seed=0)
    assert set(np.unique(env.rewards)) == {0.0, 1.0}
    np.testing.assert_allclose(env.rewards.mean(axis=0), [0.1, 0.5, 0.9], atol=0.01)


def test_pull_regret_is_zero_for_best_arm_and_positive_otherwise():
    env = _toy_env()
    for t in range(20):
        best = int(env.probs[t].argmax())
        assert env.pull(t, best)[1] == 0.0
        worst = int(env.probs[t].argmin())
        assert env.pull(t, worst)[1] == pytest.approx(env.probs[t].max() - env.probs[t].min())


def test_probabilities_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        BanditEnv.from_probs(np.zeros((2, 1)), np.array([[0.2, 1.3], [0.1, 0.4]]), seed=0)


def test_top_values_skips_reserved_ids_and_reports_coverage():
    col = np.array([3, 3, 3, 60, 60, 61, 70, 70, 70, 70])
    arms, coverage = top_values(col, 2)
    assert arms.tolist() == [70, 60] and coverage == pytest.approx(0.6)


def test_arm_probs_in_unit_interval_and_depend_on_the_arm():
    torch.manual_seed(0)
    model = DCNv2(n_ids=200, embed_dim=4, n_cross=1, hidden=(8,))
    dense, sparse = torch.randn(6, 26), torch.randint(0, 200, (6, 26))
    before = sparse.clone()
    p = arm_probs(model, 1.0, 0.0, dense, sparse, field=5, arms=np.array([60, 61, 62]))
    assert p.shape == (6, 3) and (p > 0).all() and (p < 1).all()
    assert not np.allclose(p[:, 0], p[:, 1])
    torch.testing.assert_close(sparse, before)


def test_embedding_features_drop_the_arm_field_slot():
    torch.manual_seed(0)
    model = DeepFM(n_ids=200, embed_dim=4, hidden=(8,))
    dense = np.random.default_rng(0).normal(size=(5, 26)).astype(np.float32)
    sparse = np.random.default_rng(1).integers(0, 200, (5, 26))
    sparse2 = sparse.copy()
    sparse2[:, 5] = (sparse2[:, 5] + 1) % 200
    f1 = embedding_features(model, dense, sparse, drop_field=5)
    f2 = embedding_features(model, dense, sparse2, drop_field=5)
    assert f1.shape == (5, 51 * 4)
    np.testing.assert_array_equal(f1, f2)


def test_whitened_pca_gives_unit_uncorrelated_features():
    x = np.random.default_rng(0).normal(size=(5000, 10)) @ np.random.default_rng(1).normal(
        size=(10, 10))
    z = WhitenedPCA.fit(x, 4).transform(x)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(4), atol=1e-8)


def _linear_toy(seed: int = 0, t: int = 4000, k: int = 5, d: int = 6) -> BanditEnv:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, d))
    x[:, -1] = 1.0
    theta = rng.normal(size=(k, d))
    return BanditEnv.from_probs(x, np.clip(0.5 + 0.1 * x @ theta.T, 0, 1), seed)


def _cumulative_regret(env: BanditEnv, policy) -> np.ndarray:
    regret = np.empty(env.horizon)
    for t in range(env.horizon):
        x = env.context(t)
        arm = policy.select(x)
        reward, regret[t] = env.pull(t, arm)
        policy.update(x, arm, reward)
    return np.cumsum(regret)


def test_linucb_beats_random_and_is_sublinear_on_a_linear_toy():
    env = _linear_toy()
    lin = _cumulative_regret(env, LinUCB(env.n_arms, env.features.shape[1], seed=0, alpha=0.5))
    rnd = _cumulative_regret(env, RandomPolicy(env.n_arms, seed=0))
    assert lin[-1] < 0.25 * rnd[-1]
    third = env.horizon // 3
    first, last = lin[third], lin[-1] - lin[-third - 1]
    assert last < 0.5 * first


def test_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(0)
    pol = LinUCB(n_arms=2, dim=4, seed=0, lam=2.0)
    xs = rng.normal(size=(30, 4))
    for x in xs:
        pol.update(x, 1, 1.0)
    np.testing.assert_allclose(pol.a_inv[1], np.linalg.inv(2.0 * np.eye(4) + xs.T @ xs), atol=1e-10)
    np.testing.assert_allclose(pol.a_inv[0], np.eye(4) / 2.0)


def test_eps_greedy_finds_the_best_fixed_arm():
    probs = np.tile([0.2, 0.25, 0.6, 0.3], (3000, 1))
    env = BanditEnv.from_probs(np.zeros((3000, 1)), probs, seed=0)
    pol = EpsilonGreedy(4, seed=0, eps=0.05)
    _cumulative_regret(env, pol)
    assert pol.counts.argmax() == 2
