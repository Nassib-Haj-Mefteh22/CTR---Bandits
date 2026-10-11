import torch
from torch import nn

from ctr.models.deepfm import DeepFM
from ctr.models.fm import FM, FieldEmbedding, fm_interaction


def _random_batch(n: int, n_ids: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 26, generator=g), torch.randint(0, n_ids, (n, 26), generator=g)


def _naive_pairs(v: torch.Tensor) -> torch.Tensor:
    n = v.shape[1]
    out = torch.zeros(v.shape[0], dtype=v.dtype)
    for i in range(n):
        for j in range(i + 1, n):
            out += (v[:, i] * v[:, j]).sum(dim=1)
    return out


def test_fm_trick_equals_naive_double_loop():
    v = torch.randn(5, 52, 16, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    torch.testing.assert_close(fm_interaction(v), _naive_pairs(v))


def test_field_embedding_shape_and_dense_scaling():
    emb = FieldEmbedding(n_ids=100, dense_dim=26, embed_dim=8)
    dense, sparse = _random_batch(4, 100)
    v = emb(dense, sparse)
    assert v.shape == (4, 52, 8)
    torch.testing.assert_close(v[:, 0], dense[:, :1] * emb.dense[0])
    torch.testing.assert_close(v[:, 26:], emb.sparse(sparse))


def test_fm_with_zero_vectors_is_exactly_lr():
    model = FM(n_ids=100, embed_dim=8)
    nn.init.zeros_(model.embedding.sparse.weight)
    nn.init.zeros_(model.embedding.dense)
    dense, sparse = _random_batch(4, 100)
    torch.testing.assert_close(model(dense, sparse), model.linear(dense, sparse))


def test_deepfm_shape_and_shared_embedding_gets_both_gradients():
    torch.manual_seed(0)
    model = DeepFM(n_ids=100, embed_dim=8, hidden=(16, 8))
    dense, sparse = _random_batch(4, 100)
    out = model(dense, sparse)
    assert out.shape == (4,)
    assert sum(isinstance(m, nn.Embedding) and m.embedding_dim == 8 for m in model.modules()) == 1
    model.head(model.deep(model.embedding(dense, sparse).flatten(1))).sum().backward()
    assert model.embedding.sparse.weight.grad.abs().sum() > 0


def test_deepfm_eval_is_deterministic_train_is_not():
    torch.manual_seed(0)
    model = DeepFM(n_ids=100, embed_dim=8, hidden=(64,), dropout=0.5)
    dense, sparse = _random_batch(32, 100)
    model.eval()
    torch.testing.assert_close(model(dense, sparse), model(dense, sparse))
    model.train()
    assert not torch.allclose(model(dense, sparse), model(dense, sparse))
