"""Factorization machine (Rendle 2010): logistic regression + every pairwise interaction.

logit = LR(x) + sum_{i<j} <v_i, v_j>, where v_i is a learned k-dim vector per feature.
Why vectors instead of one weight per pair: 2^20 ids give ~5e11 possible pairs and most never
co-occur in train. With W_ij = <v_i, v_j>, a pair never seen together still gets a sensible
weight, because v_i and v_j were each learned from all the OTHER pairs they appeared in.

The 52 "fields" seen by the interaction: 26 sparse fields (vector of the active hashed id) and
26 dense columns (one vector per column, scaled by the column value: x_j * v_j, as in Rendle's
real-valued FM), so numeric features also interact with categorical ones.
"""

import torch
from torch import nn

from ctr.config import EMBED_DIM, HASH_SIZE
from ctr.data import DENSE_DIM
from ctr.models.logreg import LogisticRegression

INIT_STD = 0.01  # small init: interactions start near 0, so the model starts close to LR


class FieldEmbedding(nn.Module):
    """(dense [B, D], sparse [B, F]) -> one k-dim vector per field [B, D + F, k].

    Shared by FM, DeepFM and DCN-v2: every model after LR reads its input through this layer,
    so architecture comparisons do not mix in a different input representation.
    """

    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.sparse = nn.Embedding(n_ids, embed_dim)
        nn.init.normal_(self.sparse.weight, std=INIT_STD)
        self.dense = nn.Parameter(torch.randn(dense_dim, embed_dim) * INIT_STD)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        dense_vectors = dense.unsqueeze(2) * self.dense  # [B, D, 1] * [D, k] -> [B, D, k]
        return torch.cat([dense_vectors, self.sparse(sparse)], dim=1)


def fm_interaction(v: torch.Tensor) -> torch.Tensor:
    """sum_{i<j} <v_i, v_j> for v [B, n, k] -> [B], in O(n k) instead of O(n^2 k).

    Per dimension, (sum_i v_i)^2 = sum_i v_i^2 + 2 sum_{i<j} v_i v_j, so the pair sum is
    half of (square of the sum - sum of the squares), then summed over the k dimensions.
    """
    square_of_sum = v.sum(dim=1).pow(2)  # [B, k]
    sum_of_square = v.pow(2).sum(dim=1)  # [B, k]
    return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)


class FM(nn.Module):
    """Returns LOGITS [B]. First-order part IS the Step 2 LR, so FM - LR isolates the pairs."""

    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.linear = LogisticRegression(n_ids, dense_dim)
        self.embedding = FieldEmbedding(n_ids, dense_dim, embed_dim)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        return self.linear(dense, sparse) + fm_interaction(self.embedding(dense, sparse))
