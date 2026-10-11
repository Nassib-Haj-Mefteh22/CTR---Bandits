"""Semi-synthetic contextual bandit built on the Criteo CTR models (Step 6).

Design (chosen in Step 6.1, evidence in scripts/arm_field_scan.py):
- context = one TEST row (never used to train the oracle);
- arms = the 10 most frequent train values of field C6 (the "ad" we choose to show);
- true click probability p(x, a) = Platt-calibrated DCN-v2 (seed 42) on the row with C6 set to a;
- reward = Bernoulli(p(x, a)); regret uses p directly (pseudo-regret: max_a p - p of chosen arm),
  which removes the coin-flip noise from the curves without changing their expectation;
- what POLICIES see: 32 whitened PCA components of a DIFFERENT model's (DeepFM) frozen field
  embeddings with the C6 slot removed, plus a constant 1. The policy never sees the oracle, so a
  linear model on these features is misspecified, as in production.

Limitations (logged): field semantics are unknown (Criteo anonymizes them), so "C6 = ad" is an
assumption; setting C6 to a value the row never had asks the model to extrapolate (top-10 values
cover 89% of train rows, so mostly in-distribution); the oracle is a model, not real users.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ctr.calibrate import load_trained, platt_params
from ctr.config import SEED, SPARSE_FIELDS
from ctr.data import DENSE_DIM, N_SPARSE, load_split
from ctr.train import ModelConfig

ARM_FIELD = "C6"
N_ARMS = 10
N_COMPONENTS = 32
PCA_FIT_ROWS = 200_000
N_RESERVED = 2 * N_SPARSE  # ids 0-51 = missing / rare per field: not real values, never arms
BATCH = 65536


@dataclass
class BanditEnv:
    """Fixed sequence of T rounds: features x_t [T, d], true probabilities p [T, K], and the
    pre-drawn Bernoulli rewards [T, K].

    Why pre-draw every arm's reward: every policy run with the same seed then faces the SAME
    contexts and the SAME coin flips (common random numbers), so differences between policies are
    not luck of the draw.
    """

    features: np.ndarray
    probs: np.ndarray
    rewards: np.ndarray

    @classmethod
    def from_probs(cls, features: np.ndarray, probs: np.ndarray, seed: int) -> "BanditEnv":
        if probs.min() < 0 or probs.max() > 1:
            raise ValueError("probabilities must lie in [0, 1]")
        rng = np.random.default_rng(seed)
        rewards = (rng.random(probs.shape) < probs).astype(np.float64)
        return cls(features=features, probs=probs, rewards=rewards)

    @property
    def horizon(self) -> int:
        return len(self.probs)

    @property
    def n_arms(self) -> int:
        return self.probs.shape[1]

    def context(self, t: int) -> np.ndarray:
        return self.features[t]

    def pull(self, t: int, arm: int) -> tuple[float, float]:
        """(observed reward, pseudo-regret) of playing `arm` at round t."""
        return self.rewards[t, arm], float(self.probs[t].max() - self.probs[t, arm])


def top_values(train_col: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """K most frequent real ids of one field in train and the share of train rows they cover."""
    ids, counts = np.unique(train_col[train_col >= N_RESERVED], return_counts=True)
    order = np.argsort(-counts)[:k]
    return ids[order], float(counts[order].sum() / len(train_col))


@torch.no_grad()
def arm_probs(model: nn.Module, a: float, b: float, dense: torch.Tensor, sparse: torch.Tensor,
              field: int, arms: np.ndarray) -> np.ndarray:
    """p [N, K]: Platt-calibrated sigmoid(a z + b) of every row under every arm value."""
    model.eval()
    out = []
    for arm in arms:
        s = sparse.clone()
        s[:, field] = int(arm)
        out.append(torch.sigmoid(a * model(dense, s).double() + b).numpy())
    return np.stack(out, axis=1)


@torch.no_grad()
def embedding_features(model: nn.Module, dense: np.ndarray, sparse: np.ndarray,
                       drop_field: int) -> np.ndarray:
    """Flattened frozen field embeddings [N, 51 * k] with the arm field's slot removed.

    Why remove it: the policy chooses that field; leaving the row's original value in the
    context would leak which ad the logged user actually saw.
    """
    model.eval()
    slot = DENSE_DIM + drop_field  # FieldEmbedding order: 26 dense slots, then the 26 sparse
    out = []
    for i in range(0, len(dense), BATCH):
        v = model.embedding(torch.from_numpy(dense[i:i + BATCH]),
                            torch.from_numpy(sparse[i:i + BATCH]))
        keep = torch.cat([v[:, :slot], v[:, slot + 1:]], dim=1)
        out.append(keep.flatten(1).double().numpy())
    return np.concatenate(out)


@dataclass
class WhitenedPCA:
    """x -> (x - mean) @ components / sqrt(eigenvalues): uncorrelated, unit-variance features.

    Why whiten: LinUCB's ridge prior lambda * I treats every direction alike, which only makes
    sense when features share one scale.
    """

    mean: np.ndarray
    components: np.ndarray  # [D, n]
    scale: np.ndarray  # [n]

    @classmethod
    def fit(cls, x: np.ndarray, n: int) -> "WhitenedPCA":
        mean = x.mean(axis=0)
        cov = np.cov(x - mean, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)  # ascending order
        top = np.argsort(eigval)[::-1][:n]
        return cls(mean=mean, components=eigvec[:, top], scale=np.sqrt(eigval[top]))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components / self.scale


class CriteoBanditFactory:
    """Loads the oracle, the feature model, the arms and the PCA ONCE; makes one env per seed."""

    def __init__(self, oracle_seed: int = SEED) -> None:
        self.field = SPARSE_FIELDS.index(ARM_FIELD)
        self.oracle = load_trained("dcnv2", oracle_seed, ModelConfig())
        self.a, self.b = platt_params("dcnv2", oracle_seed)
        self.feature_model = load_trained("deepfm", oracle_seed, ModelConfig())
        train_dense, train_sparse, _ = load_split("train")
        self.arms, self.coverage = top_values(train_sparse[:, self.field], N_ARMS)
        rng = np.random.default_rng(oracle_seed)
        n_fit = min(PCA_FIT_ROWS, len(train_dense))  # small samples (smoke runs) have fewer rows
        rows = rng.choice(len(train_dense), n_fit, replace=False)  # train only: no leak
        emb = embedding_features(self.feature_model, train_dense[rows], train_sparse[rows],
                                 self.field)
        self.pca = WhitenedPCA.fit(emb, N_COMPONENTS)
        self.test_dense, self.test_sparse, _ = load_split("test")

    def make_env(self, seed: int, horizon: int) -> BanditEnv:
        rows = np.random.default_rng(seed).choice(len(self.test_dense), horizon, replace=False)
        dense, sparse = self.test_dense[rows], self.test_sparse[rows]
        z = self.pca.transform(embedding_features(self.feature_model, dense, sparse, self.field))
        features = np.hstack([z, np.ones((horizon, 1))])  # constant 1 = per-arm intercept
        probs = np.concatenate([
            arm_probs(self.oracle, self.a, self.b, torch.from_numpy(dense[i:i + BATCH]),
                      torch.from_numpy(sparse[i:i + BATCH]), self.field, self.arms)
            for i in range(0, horizon, BATCH)
        ])
        return BanditEnv.from_probs(features, probs, seed)
