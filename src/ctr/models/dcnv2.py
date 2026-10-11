"""DCN-v2 (Wang et al. 2021): explicit feature crosses of bounded degree, then a deep tower.

Cross layer: x_{l+1} = x0 * (W x_l + b) + x_l, with x0 = concat of the 52 field embeddings.
Why: each layer multiplies element-wise by x0, so after L layers the output contains every
polynomial term of x0 up to degree L+1, with learned weights. FM only has degree-2 terms with
weight <v_i, v_j>; the cross network learns a full W (bit-level, not just field-level pairs), and
the residual "+ x_l" keeps lower-degree terms, so adding a layer can never remove what the
previous one learned (W = 0, b = 0 makes the layer an identity).

Structure: "stacked" (cross network feeds the deep tower). The paper also has a "parallel"
variant and a low-rank W = U V^T; full rank is affordable here (832^2 = 0.69M weights per layer).
"""

from collections.abc import Sequence

import torch
from torch import nn

from ctr.config import EMBED_DIM, HASH_SIZE
from ctr.data import DENSE_DIM, N_SPARSE
from ctr.models.deepfm import DROPOUT, HIDDEN, mlp_tower
from ctr.models.fm import FieldEmbedding

N_CROSS = 2  # middle of the {1, 2, 3} ablation grid, like EMBED_DIM = 16 in {8, 16, 32}


class CrossLayer(nn.Module):
    """x0, x_l [B, d] -> x0 * (W x_l + b) + x_l [B, d]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    """Returns LOGITS [B]. No separate LR term: the cross residuals already carry degree-1 terms."""

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
        x0 = self.embedding(dense, sparse).flatten(1)  # [B, 52 * k]
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(self.deep(x)).squeeze(1)
