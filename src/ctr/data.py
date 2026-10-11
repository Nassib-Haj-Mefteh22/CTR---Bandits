import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from ctr.config import (
    COLUMNS,
    DENSE_FIELDS,
    HASH_SIZE,
    LABEL,
    MIN_COUNT,
    PROCESSED_DIR,
    SAMPLE_FILE,
    SEED,
    SPARSE_FIELDS,
    SPLIT_FRACTIONS,
)

N_DENSE = len(DENSE_FIELDS)
N_SPARSE = len(SPARSE_FIELDS)
DENSE_DIM = 2 * N_DENSE
SPLITS = ("train", "val", "test")

_MIX_A = np.uint64(0x9E3779B97F4A7C15)
_MIX_B = np.uint64(0xC2B2AE3D27D4EB4F)


def hex_to_int(value: str) -> int:
    return int(value, 16) if value else -1


def frequent_values(
    train_values: np.ndarray, min_count: int = MIN_COUNT
) -> list[np.ndarray]:
    frequent = []
    for f in range(train_values.shape[1]):
        values, counts = np.unique(train_values[:, f], return_counts=True)
        frequent.append(values[(counts >= min_count) & (values >= 0)])
    return frequent


def rare_mask(values: np.ndarray, frequent: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(values.shape, dtype=bool)
    for f, keep in enumerate(frequent):
        mask[:, f] = (values[:, f] >= 0) & ~np.isin(values[:, f], keep, assume_unique=False)
    return mask


def hash_sparse(
    values: np.ndarray, rare: np.ndarray | None = None, hash_size: int = HASH_SIZE
) -> np.ndarray:
    n_fields = values.shape[1]
    field = np.arange(1, n_fields + 1, dtype=np.uint64)[None, :]
    missing = values < 0
    v = values.clip(min=0).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = v * _MIX_A + field * _MIX_B
        h ^= h >> np.uint64(31)
        h *= _MIX_A
        h ^= h >> np.uint64(29)
    n_reserved = 2 * n_fields
    ids = (h % np.uint64(hash_size - n_reserved)).astype(np.int64) + n_reserved
    field_ids = np.broadcast_to(np.arange(n_fields, dtype=np.int64), values.shape)
    if rare is not None:
        ids = np.where(rare, field_ids + n_fields, ids)
    return np.where(missing, field_ids, ids)


def rehash_ids(ids: np.ndarray, hash_size: int) -> np.ndarray:
    n_reserved = 2 * N_SPARSE
    folded = (ids - n_reserved) % (hash_size - n_reserved) + n_reserved
    return np.where(ids < n_reserved, ids, folded)


def transform_dense(raw: np.ndarray) -> np.ndarray:
    missing = np.isnan(raw)
    logged = np.log1p(np.where(missing, 0.0, np.clip(raw, 0.0, None)))
    return np.concatenate([logged, missing.astype(np.float64)], axis=1).astype(np.float32)


def read_sample(path=SAMPLE_FILE, nrows: int | None = None, chunksize: int = 500_000):
    converters = {f: hex_to_int for f in SPARSE_FIELDS}
    dense_parts, sparse_parts, label_parts = [], [], []
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype={LABEL: np.float32, **{f: np.float64 for f in DENSE_FIELDS}},
        converters=converters,
        nrows=nrows,
        chunksize=chunksize,
    )
    for chunk in reader:
        dense_parts.append(transform_dense(chunk[DENSE_FIELDS].to_numpy(np.float64)))
        sparse_parts.append(chunk[SPARSE_FIELDS].to_numpy(np.int64))
        label_parts.append(chunk[LABEL].to_numpy(np.float32))
    return np.concatenate(dense_parts), np.concatenate(sparse_parts), np.concatenate(label_parts)


def split_bounds(n: int, fractions=SPLIT_FRACTIONS) -> list[tuple[int, int]]:
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])
    return [(0, train_end), (train_end, val_end), (val_end, n)]


def standardize(dense_splits: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    mean = dense_splits[0].mean(axis=0)
    std = dense_splits[0].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return [((d - mean) / std).astype(np.float32) for d in dense_splits], mean, std


def build_processed(nrows: int | None = None) -> None:
    dense, raw_sparse, label = read_sample(nrows=nrows)
    bounds = split_bounds(len(label))
    frequent = frequent_values(raw_sparse[bounds[0][0] : bounds[0][1]])
    rare = rare_mask(raw_sparse, frequent)
    sparse = hash_sparse(raw_sparse, rare)
    del raw_sparse
    dense_splits, mean, std = standardize([dense[a:b] for a, b in bounds])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, (a, b), d in zip(SPLITS, bounds, dense_splits, strict=True):
        np.save(PROCESSED_DIR / f"{name}_dense.npy", d)
        np.save(PROCESSED_DIR / f"{name}_sparse.npy", sparse[a:b])
        np.save(PROCESSED_DIR / f"{name}_label.npy", label[a:b])
        print(f"{name:5s} rows {b - a:>9,}  CTR {label[a:b].mean():.4f}")
    np.save(PROCESSED_DIR / "dense_mean_std.npy", np.stack([mean, std]))
    n_frequent = sum(len(v) for v in frequent)
    print(f"frequent values (count >= {MIN_COUNT} in train): {n_frequent:,}")
    for name, (a, b) in zip(SPLITS, bounds, strict=True):
        print(f"{name:5s} share of present values mapped to 'rare': {rare[a:b].mean():.3f}")
    used = np.unique(sparse[bounds[0][0] : bounds[0][1]]).size
    print(f"distinct hashed ids in train: {used:,} of {HASH_SIZE:,} ({used / HASH_SIZE:.1%})")
    print(f"missing rate per dense field: {np.round(dense[:, N_DENSE:].mean(axis=0), 2)}")


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = ("dense", "sparse", "label")
    return tuple(np.load(PROCESSED_DIR / f"{name}_{part}.npy") for part in parts)


class CTRDataset(Dataset):
    def __init__(self, dense: np.ndarray, sparse: np.ndarray, label: np.ndarray) -> None:
        self.dense = torch.from_numpy(dense)
        self.sparse = torch.from_numpy(sparse)
        self.label = torch.from_numpy(label)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dense[idx], self.sparse[idx], self.label[idx]


def make_loader(
    dataset: CTRDataset, batch_size: int, shuffle: bool, seed: int = SEED
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    base = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None)


if __name__ == "__main__":
    build_processed()
    ds = CTRDataset(*load_split("train"))
    dense_b, sparse_b, label_b = next(iter(make_loader(ds, batch_size=4096, shuffle=True)))
    print(f"batch: dense {tuple(dense_b.shape)} {dense_b.dtype}, sparse {tuple(sparse_b.shape)} "
          f"{sparse_b.dtype}, label {tuple(label_b.shape)} {label_b.dtype}")
    print(f"sparse id range in batch: [{sparse_b.min().item()}, {sparse_b.max().item()}]")
