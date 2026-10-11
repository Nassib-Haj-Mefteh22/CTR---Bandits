import numpy as np
import torch

from ctr.data import (
    CTRDataset,
    frequent_values,
    hash_sparse,
    hex_to_int,
    make_loader,
    rare_mask,
    split_bounds,
    standardize,
    transform_dense,
)


def test_hex_to_int_and_missing():
    assert hex_to_int("68fd1e64") == 0x68FD1E64
    assert hex_to_int("") == -1


def test_hash_is_deterministic_and_in_range():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2**32, size=(1000, 26), dtype=np.int64)
    a, b = hash_sparse(raw, hash_size=1000), hash_sparse(raw.copy(), hash_size=1000)
    assert np.array_equal(a, b)
    assert a.min() >= 52 and a.max() < 1000  # real values never use the reserved ids 0..51


def test_same_value_in_different_fields_gets_different_ids():
    raw = np.full((1, 26), 12345, dtype=np.int64)
    ids = hash_sparse(raw)
    assert len(set(ids[0].tolist())) > 20  # almost surely all distinct; collisions are rare


def test_missing_gets_reserved_field_id():
    raw = np.full((2, 26), -1, dtype=np.int64)
    assert np.array_equal(hash_sparse(raw)[0], np.arange(26))


def test_transform_dense_log_clip_and_flags():
    raw = np.array([[np.nan, -2.0, 0.0, np.e - 1]])
    out = transform_dense(raw)
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :4], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(out[0, 4:], [1.0, 0.0, 0.0, 0.0])


def test_split_bounds_are_contiguous_and_cover_all():
    bounds = split_bounds(1000)
    assert bounds == [(0, 800), (800, 900), (900, 1000)]


def test_standardize_uses_train_stats_only():
    train = np.array([[0.0], [2.0]], dtype=np.float32)
    test = np.array([[100.0]], dtype=np.float32)
    (tr, te), mean, std = standardize([train, test])
    assert mean[0] == 1.0 and std[0] == 1.0
    assert te[0, 0] == 99.0  # test was NOT used to compute mean/std


def test_loader_yields_full_batches_with_right_dtypes():
    n = 10
    ds = CTRDataset(
        np.zeros((n, 26), np.float32), np.zeros((n, 26), np.int64), np.ones(n, np.float32)
    )
    batches = list(make_loader(ds, batch_size=4, shuffle=True))
    assert [len(b[2]) for b in batches] == [4, 4, 2]
    dense, sparse, label = batches[0]
    assert dense.dtype == torch.float32 and sparse.dtype == torch.int64 and label.shape == (4,)


def test_rare_values_fitted_on_train_map_to_field_rare_id():
    train = np.array([[5, 7]] * 10 + [[6, 7]], dtype=np.int64)  # 5 and 7 frequent, 6 rare
    frequent = frequent_values(train, min_count=10)
    new = np.array([[5, 9], [6, -1]], dtype=np.int64)  # 9 unseen in train, -1 missing
    mask = rare_mask(new, frequent)
    assert mask.tolist() == [[False, True], [True, False]]
    ids = hash_sparse(new, mask)
    assert ids[0, 0] >= 4  # frequent value: hashed (2 fields -> 4 reserved ids)
    assert ids[0, 1] == 2 + 1  # rare in field 1 -> n_fields + 1
    assert ids[1, 0] == 2 + 0  # rare in field 0
    assert ids[1, 1] == 1  # missing in field 1 -> 1
