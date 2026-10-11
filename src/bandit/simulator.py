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
N_RESERVED = 2 * N_SPARSE
BATCH = 65536


@dataclass
class BanditEnv:
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
        return self.rewards[t, arm], float(self.probs[t].max() - self.probs[t, arm])


def top_values(train_col: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    ids, counts = np.unique(train_col[train_col >= N_RESERVED], return_counts=True)
    order = np.argsort(-counts)[:k]
    return ids[order], float(counts[order].sum() / len(train_col))


@torch.no_grad()
def arm_probs(model: nn.Module, a: float, b: float, dense: torch.Tensor, sparse: torch.Tensor,
              field: int, arms: np.ndarray) -> np.ndarray:
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
    model.eval()
    slot = DENSE_DIM + drop_field
    out = []
    for i in range(0, len(dense), BATCH):
        v = model.embedding(torch.from_numpy(dense[i:i + BATCH]),
                            torch.from_numpy(sparse[i:i + BATCH]))
        keep = torch.cat([v[:, :slot], v[:, slot + 1:]], dim=1)
        out.append(keep.flatten(1).double().numpy())
    return np.concatenate(out)


@dataclass
class WhitenedPCA:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, n: int) -> "WhitenedPCA":
        mean = x.mean(axis=0)
        cov = np.cov(x - mean, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        top = np.argsort(eigval)[::-1][:n]
        return cls(mean=mean, components=eigvec[:, top], scale=np.sqrt(eigval[top]))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components / self.scale


class CriteoBanditFactory:
    def __init__(self, oracle_seed: int = SEED) -> None:
        self.field = SPARSE_FIELDS.index(ARM_FIELD)
        self.oracle = load_trained("dcnv2", oracle_seed, ModelConfig())
        self.a, self.b = platt_params("dcnv2", oracle_seed)
        self.feature_model = load_trained("deepfm", oracle_seed, ModelConfig())
        train_dense, train_sparse, _ = load_split("train")
        self.arms, self.coverage = top_values(train_sparse[:, self.field], N_ARMS)
        rng = np.random.default_rng(oracle_seed)
        n_fit = min(PCA_FIT_ROWS, len(train_dense))
        rows = rng.choice(len(train_dense), n_fit, replace=False)
        emb = embedding_features(self.feature_model, train_dense[rows], train_sparse[rows],
                                 self.field)
        self.pca = WhitenedPCA.fit(emb, N_COMPONENTS)
        self.test_dense, self.test_sparse, _ = load_split("test")

    def make_env(self, seed: int, horizon: int) -> BanditEnv:
        rows = np.random.default_rng(seed).choice(len(self.test_dense), horizon, replace=False)
        dense, sparse = self.test_dense[rows], self.test_sparse[rows]
        z = self.pca.transform(embedding_features(self.feature_model, dense, sparse, self.field))
        features = np.hstack([z, np.ones((horizon, 1))])
        probs = np.concatenate([
            arm_probs(self.oracle, self.a, self.b, torch.from_numpy(dense[i:i + BATCH]),
                      torch.from_numpy(sparse[i:i + BATCH]), self.field, self.arms)
            for i in range(0, horizon, BATCH)
        ])
        return BanditEnv.from_probs(features, probs, seed)
