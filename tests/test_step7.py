import numpy as np

from bandit.policies import BayesLinear, LinearTS, LinUCB, NeuralLinear, RandomPolicy
from bandit.run import run_policy
from bandit.simulator import BanditEnv


def _quadratic_toy(seed: int = 0, t: int = 4000, k: int = 4, d: int = 4) -> BanditEnv:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(k, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = rng.normal(size=(t, d))
    probs = np.clip(0.1 + 0.2 * (x @ u.T) ** 2, 0, 1)
    return BanditEnv.from_probs(np.hstack([x, np.ones((t, 1))]), probs, seed)


def test_posterior_mean_is_ridge_and_b_matches_closed_form():
    rng = np.random.default_rng(0)
    phi, r = rng.normal(size=(50, 3)), rng.normal(size=50)
    bayes = BayesLinear(n_arms=2, dim=3, lam=2.0, a0=6.0, b0=1.0)
    for f, y in zip(phi, r, strict=True):
        bayes.add(1, f, y)
    precision = 2.0 * np.eye(3) + phi.T @ phi
    mu = np.linalg.solve(precision, phi.T @ r)
    np.testing.assert_allclose(bayes.mu[1], mu, atol=1e-10)
    np.testing.assert_allclose(bayes.b[1], 1.0 + 0.5 * (r @ r - mu @ precision @ mu), atol=1e-9)
    np.testing.assert_allclose(bayes.chol_cov[1] @ bayes.chol_cov[1].T, np.linalg.inv(precision),
                               atol=1e-10)
    assert bayes.n[0] == 0 and np.allclose(bayes.mu[0], 0)


def test_rebuild_equals_incremental_updates():
    rng = np.random.default_rng(1)
    arms, phi, r = rng.integers(0, 3, 40), rng.normal(size=(40, 4)), rng.random(40)
    inc, reb = BayesLinear(3, 4), BayesLinear(3, 4)
    for a, f, y in zip(arms, phi, r, strict=True):
        inc.add(int(a), f, y)
    reb.rebuild(arms, phi, r)
    for attr in ("precision", "mu", "b", "n"):
        np.testing.assert_allclose(getattr(reb, attr), getattr(inc, attr), atol=1e-10)


def test_thompson_samples_concentrate_with_data():
    rng = np.random.default_rng(0)
    bayes = BayesLinear(1, 2)
    phi = np.array([1.0, 0.5])
    spread_before = np.std([bayes.sample_scores(phi, rng)[0] for _ in range(500)])
    for _ in range(2000):
        bayes.add(0, rng.normal(size=2), rng.normal())
    spread_after = np.std([bayes.sample_scores(phi, rng)[0] for _ in range(500)])
    assert spread_after < 0.2 * spread_before


def test_neural_linear_beats_linucb_on_a_nonlinear_toy():
    env = _quadratic_toy()
    k, d = env.n_arms, env.features.shape[1]
    neural = run_policy(env, NeuralLinear(k, d, seed=0, retrain_every=250, steps=100))[-1]
    lin = run_policy(env, LinUCB(k, d, seed=0, alpha=0.25))[-1]
    rnd = run_policy(env, RandomPolicy(k, seed=0))[-1]
    assert neural < 0.7 * lin
    assert neural < 0.5 * rnd


def test_linear_ts_learns_on_a_linear_toy():
    rng = np.random.default_rng(0)
    x = np.hstack([rng.normal(size=(3000, 4)), np.ones((3000, 1))])
    probs = np.clip(0.5 + 0.1 * x @ rng.normal(size=(5, 5)).T, 0, 1)
    env = BanditEnv.from_probs(x, probs, seed=0)
    ts = run_policy(env, LinearTS(5, 5, seed=0))
    rnd = run_policy(env, RandomPolicy(5, seed=0))
    assert ts[-1] < 0.3 * rnd[-1]
