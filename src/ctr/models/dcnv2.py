from collections.abc import Sequence

import torch
from torch import nn

from ctr.config import EMBED_DIM, HASH_SIZE
from ctr.data import DENSE_DIM, N_SPARSE
from ctr.models.deepfm import DROPOUT, HIDDEN, mlp_tower
from ctr.models.fm import FieldEmbedding

N_CROSS = 2


class CrossLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM, n_cross: int = N_CROSS,
                 hidden: Sequence[int] = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.embedding = FieldEmbedding(n_ids, dense_dim, embed_dim)
        dim = (dense_dim + N_SPARSE) * embed_dim
        self.cross = nn.ModuleList(CrossLayer(dim) for _ in range(n_cross))
        self.deep = mlp_tower(dim, hidden, dropout)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(dense, sparse).flatten(1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(self.deep(x)).squeeze(1)
