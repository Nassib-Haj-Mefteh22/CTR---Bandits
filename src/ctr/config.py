"""Single place for paths, seeds and data settings.

Why: every script reads the same constants, so an experiment is defined by this file plus
the command line, and changing one setting cannot silently drift between scripts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Criteo Display Advertising Challenge layout:
# label, 13 integer features, 26 categorical (hex) features.
DENSE_FIELDS: list[str] = [f"I{i}" for i in range(1, 14)]
SPARSE_FIELDS: list[str] = [f"C{i}" for i in range(1, 27)]
LABEL = "label"
COLUMNS: list[str] = [LABEL, *DENSE_FIELDS, *SPARSE_FIELDS]

SEED = 42
SAMPLE_ROWS = 3_000_000  # first N rows of train.txt: fits CPU training in minutes per epoch
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train/val/test by row order (roughly chronological)
# ONE hash space shared by all 26 fields (field name is mixed into the hash). Per-field 2**20
# tables would be 26 * 2**20 * 16 floats = 1.7 GB of embeddings: too big for CPU.
HASH_SIZE = 2**20
MIN_COUNT = 10  # categorical values seen < 10 times in train -> one 'rare' id per field
EMBED_DIM = 16  # k: size of each id's vector in FM / DeepFM / DCN-v2 (Step 4 ablates 8/16/32)

RAW_TRAIN_FILE = RAW_DIR / "train.txt"
SAMPLE_FILE = RAW_DIR / "sample.tsv"


if __name__ == "__main__":
    print(f"root          {ROOT}")
    print(f"sample file   {SAMPLE_FILE} (exists: {SAMPLE_FILE.exists()})")
    print(f"fields        {len(DENSE_FIELDS)} dense, {len(SPARSE_FIELDS)} sparse")
    print(f"sample rows   {SAMPLE_ROWS:,}  split {SPLIT_FRACTIONS}")
    print(f"hash size     {HASH_SIZE:,}  seed {SEED}  embed dim {EMBED_DIM}")
