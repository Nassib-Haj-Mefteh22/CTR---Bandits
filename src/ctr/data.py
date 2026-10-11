"""Load the Criteo sample, turn it into model-ready arrays, split by time, and serve batches.

Pipeline: sample.tsv -> (read in chunks) -> dense: log1p + missing flags, sparse: hashed ids
-> split by row order 80/10/10 -> standardize dense with TRAIN statistics -> .npy per split
-> CTRDataset + DataLoader that return whole batches.
"""

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
DENSE_DIM = 2 * N_DENSE  # 13 log-values + 13 missing flags
SPLITS = ("train", "val", "test")

# Two large odd 64-bit constants (golden-ratio and xxhash primes) for a cheap, STABLE mix.
# Python's built-in hash() is salted per process, so ids would change between runs: unusable.
_MIX_A = np.uint64(0x9E3779B97F4A7C15)
_MIX_B = np.uint64(0xC2B2AE3D27D4EB4F)


def hex_to_int(value: str) -> int:
    """Criteo categorical values are 8-char hex strings; missing (empty) becomes -1."""
    return int(value, 16) if value else -1


def frequent_values(
    train_values: np.ndarray, min_count: int = MIN_COUNT
) -> list[np.ndarray]:
    """Per field, the sorted raw values seen at least min_count times in TRAIN.

    Why: in 2.4M train rows there are 2.67M distinct values but only 128k appear 10+ times, and
    those cover 94% of occurrences. A value seen 1-9 times cannot get a reliable embedding, and
    hashing millions of them into the table causes collisions that corrupt the frequent ones.
    Counted on train only: val/test frequencies would leak information from the future.
    """
    frequent = []
    for f in range(train_values.shape[1]):
        values, counts = np.unique(train_values[:, f], return_counts=True)
        frequent.append(values[(counts >= min_count) & (values >= 0)])
    return frequent


def rare_mask(values: np.ndarray, frequent: list[np.ndarray]) -> np.ndarray:
    """True where a present (non-missing) value is NOT frequent in train (incl. unseen values)."""
    mask = np.zeros(values.shape, dtype=bool)
    for f, keep in enumerate(frequent):
        mask[:, f] = (values[:, f] >= 0) & ~np.isin(values[:, f], keep, assume_unique=False)
    return mask


def hash_sparse(
    values: np.ndarray, rare: np.ndarray | None = None, hash_size: int = HASH_SIZE
) -> np.ndarray:
    """Map raw categorical ints [n, 26] to ids in one shared table of size hash_size.

    Id layout: 0..25 = "missing in field f", 26..51 = "rare value in field f",
    52..hash_size-1 = hashed frequent values. Reserved ids never collide with real values.
    Why mix the field index into the hash: the same raw value in C3 and C7 means different
    things, so it must land on different rows of the shared embedding table.
    """
    n_fields = values.shape[1]
    field = np.arange(1, n_fields + 1, dtype=np.uint64)[None, :]
    missing = values < 0
    v = values.clip(min=0).astype(np.uint64)
    with np.errstate(over="ignore"):  # uint64 wrap-around is the intended modular arithmetic
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
    """Fold ids from the stored 2^20 table into a smaller table, keeping the reserved ids.

    Why folding instead of re-running the pipeline: (h mod 2^20) mod m is still a hash of the raw
    value, so collisions behave like a fresh hash into m slots, and the hash-size ablation then
    reuses exactly the same rows, split and rare-value rule (only ONE variable changes).
    """
    n_reserved = 2 * N_SPARSE
    folded = (ids - n_reserved) % (hash_size - n_reserved) + n_reserved
    return np.where(ids < n_reserved, ids, folded)


def transform_dense(raw: np.ndarray) -> np.ndarray:
    """Raw counts [n, 13] (NaN = missing) -> [n, 26] = log1p(clipped counts) + missing flags.

    Why log1p: counts are heavy-tailed (values up to millions); log compresses them so a few huge
    values do not dominate the gradient. Why clip at 0: a few fields have -1/-2 values, log of a
    negative is undefined. Why flags: "missing" carries signal and 0 alone would hide it.
    """
    missing = np.isnan(raw)
    logged = np.log1p(np.where(missing, 0.0, np.clip(raw, 0.0, None)))
    return np.concatenate([logged, missing.astype(np.float64)], axis=1).astype(np.float32)


def read_sample(path=SAMPLE_FILE, nrows: int | None = None, chunksize: int = 500_000):
    """Read sample.tsv in chunks -> (dense [n,26] float32, raw sparse ints [n,26], label [n]).

    Why chunks + hex->int at read time: 3M x 26 Python strings would cost several GB of RAM;
    ints cost 8 bytes each (~0.6 GB total). Hashing happens later, after the split, because the
    rare-value rule must be fitted on train only.
    """
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
    """Contiguous [start, end) row ranges for train/val/test, in file (= time) order.

    Why not a random split: rows are chronological, a random split would let the model train on
    clicks from after the test period (leakage) and overstate AUC.
    """
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])
    return [(0, train_end), (train_end, val_end), (val_end, n)]


def standardize(dense_splits: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Scale every split with mean/std computed on TRAIN only (val/test must stay unseen)."""
    mean = dense_splits[0].mean(axis=0)
    std = dense_splits[0].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # constant column: leave it, avoid division by ~0
    return [((d - mean) / std).astype(np.float32) for d in dense_splits], mean, std


def build_processed(nrows: int | None = None) -> None:
    """sample.tsv -> data/processed/{split}_{dense,sparse,label}.npy, printing a summary."""
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
    """Load one processed split as (dense, sparse, label) arrays."""
    parts = ("dense", "sparse", "label")
    return tuple(np.load(PROCESSED_DIR / f"{name}_{part}.npy") for part in parts)


class CTRDataset(Dataset):
    """Holds one split as tensors; __getitem__ takes a LIST of indices and returns a whole batch.

    Why batch indexing: fetching 3M rows one by one through Python is the bottleneck of a CPU
    training loop; one fancy-index per batch is ~1000x fewer Python calls.
    """

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
    """DataLoader that yields (dense [B,26], sparse [B,26], label [B]) batches.

    batch_size=None turns off automatic batching: the BatchSampler hands a list of indices to
    __getitem__, which already returns the full batch. Shuffle only for training; a seeded
    generator makes the batch order reproducible.
    """
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
