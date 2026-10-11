import torch
from torch import nn

from ctr.config import HASH_SIZE
from ctr.data import DENSE_DIM


class LogisticRegression(nn.Module):
    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM) -> None:
        super().__init__()
        self.sparse_weight = nn.Embedding(n_ids, 1)
        nn.init.zeros_(self.sparse_weight.weight)
        self.dense = nn.Linear(dense_dim, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        sparse_part = self.sparse_weight(sparse).sum(dim=(1, 2))
        return self.dense(dense).squeeze(1) + sparse_part
