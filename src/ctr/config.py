from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DENSE_FIELDS: list[str] = [f"I{i}" for i in range(1, 14)]
SPARSE_FIELDS: list[str] = [f"C{i}" for i in range(1, 27)]
LABEL = "label"
COLUMNS: list[str] = [LABEL, *DENSE_FIELDS, *SPARSE_FIELDS]

SEED = 42
SAMPLE_ROWS = 3_000_000
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
HASH_SIZE = 2**20
MIN_COUNT = 10
EMBED_DIM = 16

RAW_TRAIN_FILE = RAW_DIR / "train.txt"
SAMPLE_FILE = RAW_DIR / "sample.tsv"


if __name__ == "__main__":
    print(f"root          {ROOT}")
    print(f"sample file   {SAMPLE_FILE} (exists: {SAMPLE_FILE.exists()})")
    print(f"fields        {len(DENSE_FIELDS)} dense, {len(SPARSE_FIELDS)} sparse")
    print(f"sample rows   {SAMPLE_ROWS:,}  split {SPLIT_FRACTIONS}")
    print(f"hash size     {HASH_SIZE:,}  seed {SEED}  embed dim {EMBED_DIM}")
