"""Stream the first N rows of Criteo DAC train.txt into data/raw/sample.tsv.

Why streaming: dac.tar.gz is ~4 GB compressed / ~11 GB raw, but we only need the first few
million rows. Reading the HTTP response through a streaming tar+gzip decoder ("r|gz") lets us
stop as soon as we have N rows, so only the beginning of the archive is ever downloaded.
Why the FIRST rows: the file is in chronological order, so the first N rows are a contiguous
time window and a row-order split later still means "train on the past, test on the future".

Usage: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe scripts\\sample.py [--rows 3000000]
"""

import argparse
import tarfile
import time
import urllib.request

from ctr.config import RAW_DIR, SAMPLE_FILE, SAMPLE_ROWS

DAC_URL = "https://ndownloader.figshare.com/files/10082655"  # dac.tar.gz (Kaggle DAC release)
N_COLUMNS = 40  # label + 13 dense + 26 sparse


def stream_sample(n_rows: int, url: str = DAC_URL) -> tuple[int, int]:
    """Write the first n_rows of train.txt to SAMPLE_FILE; return (rows written, clicks)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SAMPLE_FILE.with_suffix(".part")  # never leave a half-written sample.tsv behind
    rows = clicks = bad = 0
    start = time.time()
    with urllib.request.urlopen(url, timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                print(f"archive member: {member.name} ({member.size / 1e9:.2f} GB)")
                if not member.name.endswith("train.txt"):
                    continue  # skip readme/test without extracting them
                handle = archive.extractfile(member)
                assert handle is not None
                with open(tmp, "w", encoding="utf-8", newline="\n") as out:
                    for raw in handle:
                        line = raw.decode("utf-8").rstrip("\n")
                        if line.count("\t") != N_COLUMNS - 1:
                            bad += 1  # malformed row: count it, do not silently keep it
                            continue
                        out.write(line + "\n")
                        clicks += line[0] == "1"
                        rows += 1
                        if rows % 500_000 == 0:
                            print(f"  {rows:,} rows  {time.time() - start:.0f}s")
                        if rows >= n_rows:
                            break
                break
    if rows == 0:
        raise RuntimeError("train.txt not found in archive or empty")
    tmp.replace(SAMPLE_FILE)
    print(f"malformed rows skipped: {bad}")
    return rows, clicks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=SAMPLE_ROWS)
    args = parser.parse_args()
    if SAMPLE_FILE.exists():
        print(f"{SAMPLE_FILE} already exists, nothing to do (delete it to resample)")
        return
    rows, clicks = stream_sample(args.rows)
    size_mb = SAMPLE_FILE.stat().st_size / 1e6
    print(f"wrote {rows:,} rows ({size_mb:.0f} MB) to {SAMPLE_FILE}")
    print(f"CTR = {clicks / rows:.4f}")


if __name__ == "__main__":
    main()
