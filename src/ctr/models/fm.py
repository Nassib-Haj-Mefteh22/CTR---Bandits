import torch
from torch import nn

from ctr.config import EMBED_DIM, HASH_SIZE
from ctr.data import DENSE_DIM
from ctr.models.logreg import LogisticRegression

INIT_STD = 0.01


class FieldEmbedding(nn.Module):
    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.sparse = nn.Embedding(n_ids, embed_dim)
        nn.init.normal_(self.sparse.weight, std=INIT_STD)
        self.dense = nn.Parameter(torch.randn(dense_dim, embed_dim) * INIT_STD)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        dense_vectors = dense.unsqueeze(2) * self.dense
        return torch.cat([dense_vectors, self.sparse(sparse)], dim=1)


def fm_interaction(v: torch.Tensor) -> torch.Tensor:
    square_of_sum = v.sum(dim=1).pow(2)
    sum_of_square = v.pow(2).sum(dim=1)
    return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)


class FM(nn.Module):
    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.linear = LogisticRegression(n_ids, dense_dim)
        self.embedding = FieldEmbedding(n_ids, dense_dim, embed_dim)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        return self.linear(dense, sparse) + fm_interaction(self.embedding(dense, sparse))
