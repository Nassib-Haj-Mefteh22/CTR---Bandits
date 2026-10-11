import numpy as np
import torch
from torch import nn


def argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(values == values.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


class RandomPolicy:
    name = "random"

    def __init__(self, n_arms: int, seed: int) -> None:
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        pass


class EpsilonGreedy:
    name = "eps-greedy"

    def __init__(self, n_arms: int, seed: int, eps: float = 0.05) -> None:
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def select(self, x: np.ndarray) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(len(self.counts)))
        means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), np.inf)
        return argmax_random_tie(means, self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class LinUCB:
    name = "linucb"

    def __init__(self, n_arms: int, dim: int, seed: int, alpha: float = 0.25,
                 lam: float = 1.0) -> None:
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.a_inv = np.tile(np.eye(dim) / lam, (n_arms, 1, 1))
        self.b = np.zeros((n_arms, dim))

    def scores(self, x: np.ndarray) -> np.ndarray:
        theta = np.einsum("kij,kj->ki", self.a_inv, self.b)
        width = np.sqrt(np.einsum("i,kij,j->k", x, self.a_inv, x))
        return theta @ x + self.alpha * width

    def select(self, x: np.ndarray) -> int:
        return argmax_random_tie(self.scores(x), self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        a_inv_x = self.a_inv[arm] @ x
        self.a_inv[arm] -= np.outer(a_inv_x, a_inv_x) / (1.0 + x @ a_inv_x)
        self.b[arm] += reward * x


class BayesLinear:
    def __init__(self, n_arms: int, dim: int, lam: float = 1.0, a0: float = 6.0,
                 b0: float = 1.0) -> None:
        self.lam, self.a0, self.b0 = lam, a0, b0
        self.n_arms, self.dim = n_arms, dim
        self.reset()

    def reset(self) -> None:
        k, d = self.n_arms, self.dim
        self.precision = np.tile(self.lam * np.eye(d), (k, 1, 1))
        self.xty = np.zeros((k, d))
        self.yty = np.zeros(k)
        self.n = np.zeros(k)
        self.mu = np.zeros((k, d))
        self.chol_cov = np.tile(np.eye(d) / np.sqrt(self.lam), (k, 1, 1))
        self.b = np.full(k, self.b0)

    def _refresh(self, arm: int) -> None:
        cov = np.linalg.inv(self.precision[arm])
        cov = 0.5 * (cov + cov.T)
        self.mu[arm] = cov @ self.xty[arm]
        self.chol_cov[arm] = np.linalg.cholesky(cov)
        quad = self.mu[arm] @ self.precision[arm] @ self.mu[arm]
        self.b[arm] = self.b0 + 0.5 * max(self.yty[arm] - quad, 0.0)

    def add(self, arm: int, phi: np.ndarray, reward: float) -> None:
        self.precision[arm] += np.outer(phi, phi)
        self.xty[arm] += reward * phi
        self.yty[arm] += reward * reward
        self.n[arm] += 1
        self._refresh(arm)

    def rebuild(self, arms: np.ndarray, phis: np.ndarray, rewards: np.ndarray) -> None:
        self.reset()
        for arm in range(self.n_arms):
            mask = arms == arm
            f, r = phis[mask], rewards[mask]
            self.precision[arm] += f.T @ f
            self.xty[arm] = f.T @ r
            self.yty[arm] = r @ r
            self.n[arm] = mask.sum()
            self._refresh(arm)

    def sample_scores(self, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a = self.a0 + self.n / 2
        sigma2 = self.b / rng.gamma(a)
        z = rng.standard_normal((self.n_arms, self.dim))
        beta = self.mu + np.sqrt(sigma2)[:, None] * np.einsum("kij,kj->ki", self.chol_cov, z)
        return beta @ phi


class LinearTS:
    name = "linear-ts"

    def __init__(self, n_arms: int, dim: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.bayes = BayesLinear(n_arms, dim)

    def select(self, x: np.ndarray) -> int:
        return argmax_random_tie(self.bayes.sample_scores(x, self.rng), self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self.bayes.add(arm, x, reward)


class NeuralLinear:
    name = "neural-linear"

    def __init__(self, n_arms: int, dim: int, seed: int, hidden: tuple[int, int] = (64, 32),
                 retrain_every: int = 1000, steps: int = 200, batch: int = 512,
                 lr: float = 1e-3) -> None:
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.retrain_every, self.steps, self.batch = retrain_every, steps, batch
        self.body = nn.Sequential(nn.Linear(dim, hidden[0]), nn.ReLU(),
                                  nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.head = nn.Linear(hidden[1], n_arms)
        self.optimizer = torch.optim.Adam([*self.body.parameters(), *self.head.parameters()],
                                          lr=lr)
        self.bayes = BayesLinear(n_arms, hidden[1] + 1)
        self.xs: list[np.ndarray] = []
        self.arms: list[int] = []
        self.rewards: list[float] = []
        self._export()

    def _export(self) -> None:
        self.w = [(m.weight.detach().double().numpy().T.copy(), m.bias.detach().double().numpy())
                  for m in self.body if isinstance(m, nn.Linear)]

    def features(self, x: np.ndarray) -> np.ndarray:
        h = x
        for w, b in self.w:
            h = np.maximum(h @ w + b, 0.0)
        ones = np.ones(h.shape[:-1] + (1,))
        return np.concatenate([h, ones], axis=-1)

    def select(self, x: np.ndarray) -> int:
        return argmax_random_tie(self.bayes.sample_scores(self.features(x), self.rng), self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self.xs.append(x)
        self.arms.append(arm)
        self.rewards.append(reward)
        if len(self.xs) % self.retrain_every == 0:
            self._retrain()
        else:
            self.bayes.add(arm, self.features(x), reward)

    def _retrain(self) -> None:
        x = torch.from_numpy(np.asarray(self.xs, dtype=np.float32))
        arms = torch.tensor(self.arms)
        r = torch.tensor(self.rewards, dtype=torch.float32)
        gen = torch.Generator().manual_seed(int(self.rng.integers(2**31)))
        for _ in range(self.steps):
            idx = torch.randint(len(x), (self.batch,), generator=gen)
            out = self.head(self.body(x[idx])).gather(1, arms[idx, None]).squeeze(1)
            loss = ((out - r[idx]) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._export()
        self.bayes.rebuild(np.asarray(self.arms), self.features(np.asarray(self.xs)),
                           np.asarray(self.rewards))
