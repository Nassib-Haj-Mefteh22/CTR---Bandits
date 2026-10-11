"""DeepFM (Guo et al. 2017): FM + an MLP that reads the SAME field embeddings.

logit = LR(x) + FM pairs(v) + MLP(concat(v)).
Why shared embeddings: the FM part learns pairwise structure, the MLP learns higher-order,
non-linear combinations, and both push gradients into one table. Unlike Wide&Deep, no hand-made
cross features are needed, and the MLP gets embeddings already shaped by the FM signal.
"""

from collections.abc import Sequence

import torch
from torch import nn

from ctr.config import EMBED_DIM, HASH_SIZE
from ctr.data import DENSE_DIM, N_SPARSE
from ctr.models.fm import FieldEmbedding, fm_interaction
from ctr.models.logreg import LogisticRegression

HIDDEN = (256, 128)  # paper uses 400-400-400; smaller keeps CPU epochs short
DROPOUT = 0.2


def mlp_tower(in_dim: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    """Linear -> ReLU -> Dropout blocks, no output head (DCN-v2 reuses it as its deep tower)."""
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = width
    return nn.Sequential(*layers)


class DeepFM(nn.Module):
    """Returns LOGITS [B]. Dropout is only active in train mode (predict() calls model.eval())."""

    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM,
                 embed_dim: int = EMBED_DIM, hidden: Sequence[int] = HIDDEN,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.linear = LogisticRegression(n_ids, dense_dim)
        self.embedding = FieldEmbedding(n_ids, dense_dim, embed_dim)
        n_fields = dense_dim + N_SPARSE
        self.deep = mlp_tower(n_fields * embed_dim, hidden, dropout)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        v = self.embedding(dense, sparse)  # [B, 52, k], used by BOTH parts below
        deep = self.head(self.deep(v.flatten(1))).squeeze(1)
        return self.linear(dense, sparse) + fm_interaction(v) + deep
