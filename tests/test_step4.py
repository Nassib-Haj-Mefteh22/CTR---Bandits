import numpy as np
import torch
from torch import nn

from ctr.data import rehash_ids
from ctr.models.dcnv2 import CrossLayer, DCNv2
from ctr.train import ModelConfig, build_model


def _random_batch(n: int, n_ids: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 26, generator=g), torch.randint(0, n_ids, (n, 26), generator=g)


# --- 4.1 DCN-v2 --------------------------------------------------------------------------------
def test_cross_layer_with_zero_weights_is_identity():
    layer = CrossLayer(12)
    nn.init.zeros_(layer.linear.weight)
    nn.init.zeros_(layer.linear.bias)
    g = torch.Generator().manual_seed(0)
    x0, x = torch.randn(5, 12, generator=g), torch.randn(5, 12, generator=g)
    torch.testing.assert_close(layer(x0, x), x)


def test_cross_layer_matches_formula():
    torch.manual_seed(0)
    layer = CrossLayer(6)
    x0, x = torch.randn(3, 6), torch.randn(3, 6)
    expected = x0 * (x @ layer.linear.weight.T + layer.linear.bias) + x
    torch.testing.assert_close(layer(x0, x), expected)


def test_stacked_cross_layers_raise_polynomial_degree():
    """d = 1, W = w, b = 0: one layer gives w t^2 + t, two layers give a degree-3 polynomial."""
    layers = [CrossLayer(1), CrossLayer(1)]
    for layer, w in zip(layers, (2.0, 3.0), strict=True):
        nn.init.constant_(layer.linear.weight, w)
        nn.init.zeros_(layer.linear.bias)
    t = torch.linspace(-2, 2, 7).unsqueeze(1)
    x1 = layers[0](t, t)
    torch.testing.assert_close(x1, 2 * t**2 + t)
    x2 = layers[1](t, x1)  # t * 3 * (2t^2 + t) + (2t^2 + t) = 6t^3 + 5t^2 + t
    torch.testing.assert_close(x2, 6 * t**3 + 5 * t**2 + t)


def test_dcnv2_shape_and_number_of_cross_layers():
    torch.manual_seed(0)
    model = DCNv2(n_ids=100, embed_dim=4, n_cross=3, hidden=(16, 8))
    dense, sparse = _random_batch(5, 100)
    assert model(dense, sparse).shape == (5,)
    assert len(model.cross) == 3
    assert model.cross[0].linear.weight.shape == (52 * 4, 52 * 4)


def test_dcnv2_zero_cross_weights_reduces_to_embedding_plus_mlp():
    torch.manual_seed(0)
    model = DCNv2(n_ids=100, embed_dim=4, n_cross=2, hidden=(16,)).eval()
    for layer in model.cross:
        nn.init.zeros_(layer.linear.weight)
        nn.init.zeros_(layer.linear.bias)
    dense, sparse = _random_batch(5, 100)
    x0 = model.embedding(dense, sparse).flatten(1)
    torch.testing.assert_close(model(dense, sparse), model.head(model.deep(x0)).squeeze(1))


# --- 4.3 hash-size ablation --------------------------------------------------------------------
def test_rehash_keeps_reserved_ids_and_stays_in_range():
    ids = np.array([[0, 25, 26, 51, 52, 2**20 - 1, 2**18 + 7]])
    out = rehash_ids(ids, 2**18)
    assert out[0, :4].tolist() == [0, 25, 26, 51]  # missing / rare ids untouched
    assert out[0, 4] == 52 and out.min() >= 0 and out.max() < 2**18
    assert (out[0, 4:] >= 52).all()  # real values never fold onto a reserved id


def test_build_model_passes_only_relevant_knobs():
    mcfg = ModelConfig(embed_dim=4, n_cross=3, hash_size=128)
    assert build_model("logreg", mcfg).sparse_weight.num_embeddings == 128
    assert build_model("deepfm", mcfg).embedding.sparse.embedding_dim == 4
    assert len(build_model("dcnv2", mcfg).cross) == 3
