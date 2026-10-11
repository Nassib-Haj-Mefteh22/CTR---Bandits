"""Logistic regression baseline: logit = bias + sum of one weight per active sparse id + w . dense.

Why an embedding of size 1: a one-hot logistic regression over ~1M hashed ids is a weight vector
of length 1M multiplied by a vector with exactly 26 ones. Looking up those 26 weights and summing
them is the same computation without ever building the one-hot vector.
"""

import torch
from torch import nn

from ctr.config import HASH_SIZE
from ctr.data import DENSE_DIM


class LogisticRegression(nn.Module):
    """Returns LOGITS [B] (not probabilities): BCEWithLogitsLoss applies the sigmoid stably."""

    def __init__(self, n_ids: int = HASH_SIZE, dense_dim: int = DENSE_DIM) -> None:
        super().__init__()
        self.sparse_weight = nn.Embedding(n_ids, 1)
        # Zero init: every id starts with no opinion, so the initial prediction is sigmoid(bias)
        # for all rows. Random init would add noise from 26 random weights to every logit.
        nn.init.zeros_(self.sparse_weight.weight)
        self.dense = nn.Linear(dense_dim, 1)  # its bias is the global bias of the model

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """dense [B, dense_dim] float, sparse [B, n_fields] long -> logits [B]."""
        sparse_part = self.sparse_weight(sparse).sum(dim=(1, 2))  # [B, F, 1] -> [B]
        return self.dense(dense).squeeze(1) + sparse_part
