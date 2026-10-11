"""Step 4.3 ablations: change ONE knob of the best model at a time, then repeat the best config
over extra seeds. Each run appends one row to results/ablations.csv (resumable: runs already in
the csv are skipped), then a summary table and results/figures/ablations.png are written.

Run:  $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe scripts/ablate.py --model dcnv2
      add --plot-only to redraw the table and figure from the csv without training.
"""

import argparse
import csv
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ctr.config import FIGURES_DIR, RESULTS_DIR, SEED  # noqa: E402
from ctr.train import (  # noqa: E402
    METRICS_CSV,
    ModelConfig,
    TrainConfig,
    append_metrics_row,
    run,
)

ABLATIONS_CSV = RESULTS_DIR / "ablations.csv"
GRID: dict[str, list[int]] = {
    "embed_dim": [8, 16],  # k=32 dropped: ~45 CPU min for one point (sprint budget)
    "n_cross": [1, 2, 3],
    "hash_size": [2**18, 2**20],
}
EXTRA_SEEDS = [SEED + 1, SEED + 2]
# Every embedding model so far peaked at epoch 2 and got worse at 3: patience 1 stops one epoch
# earlier and selects the SAME best epoch, saving ~25% of CPU time.
PATIENCE = 1
LABELS = {"embed_dim": "embedding dim k", "n_cross": "cross layers", "hash_size": "hash size"}


def planned_runs(model: str) -> list[tuple[str, ModelConfig, int]]:
    """(varied knob, config, seed) for every run; 'baseline' = all defaults, seed 42."""
    base = ModelConfig()
    runs = [("baseline", base, SEED)]
    for knob, values in GRID.items():
        if knob == "n_cross" and model != "dcnv2":
            continue  # only DCN-v2 has cross layers
        for value in values:
            if value != getattr(base, knob):
                runs.append((knob, replace(base, **{knob: value}), SEED))
    runs += [("seed", base, s) for s in EXTRA_SEEDS]
    return runs


def done_keys() -> set[tuple]:
    if not ABLATIONS_CSV.exists():
        return set()
    with ABLATIONS_CSV.open(encoding="utf-8") as f:
        return {(r["model"], r["embed_dim"], r["n_cross"], r["hash_size"], r["seed"])
                for r in csv.DictReader(f)}


def key(model: str, mcfg: ModelConfig, seed: int) -> tuple:
    return (model, str(mcfg.embed_dim), str(mcfg.n_cross), str(mcfg.hash_size), str(seed))


def copy_baseline(model: str) -> bool:
    """Reuse the main-table run (same defaults, seed 42) as the baseline row instead of
    retraining it: same code, same seed, so a rerun would only cost ~18 CPU minutes."""
    with METRICS_CSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == model and r["seed"] == str(SEED)]
    if not rows:
        return False
    append_metrics_row({"varied": "baseline", **asdict(ModelConfig()), **rows[-1]}, ABLATIONS_CSV)
    print(f"baseline copied from {METRICS_CSV.name}")
    return True


def train_all(model: str) -> None:
    done = done_keys()
    for varied, mcfg, seed in planned_runs(model):
        if key(model, mcfg, seed) in done:
            print(f"skip (already in csv): {varied} {mcfg} seed {seed}")
            continue
        if varied == "baseline" and copy_baseline(model):
            continue
        print(f"=== {varied}: {mcfg}, seed {seed}", flush=True)
        run(model, TrainConfig(seed=seed, patience=PATIENCE), final=True, mcfg=mcfg,
            csv_path=ABLATIONS_CSV, extra={"varied": varied, **asdict(mcfg)})


def summarize(model: str) -> pd.DataFrame:
    """Print the one-knob table and the seed mean +/- std; return the rows of this model."""
    df = pd.read_csv(ABLATIONS_CSV)
    df = df[df["model"] == model]
    cols = ["varied", "embed_dim", "n_cross", "hash_size", "seed", "best_epoch",
            "val_logloss", "val_auc", "test_auc", "test_logloss", "n_params", "train_seconds"]
    print(df[cols].to_string(index=False))
    base = ModelConfig()
    same = df[(df["embed_dim"] == base.embed_dim) & (df["n_cross"] == base.n_cross)
              & (df["hash_size"] == base.hash_size)]
    print(f"\nbest config over {len(same)} seeds {sorted(same['seed'])}:")
    for m in ("val_auc", "test_auc", "test_logloss"):
        print(f"  {m:13s} mean {same[m].mean():.5f}  std {same[m].std(ddof=1):.5f}")
    return df


def plot(df: pd.DataFrame, model: str) -> None:
    """One small panel per knob (one series each): test AUC vs value, seed spread on baseline."""
    base = ModelConfig()
    knobs = [k for k in GRID if df[k].nunique() > 1]
    seeds = df[(df["embed_dim"] == base.embed_dim) & (df["n_cross"] == base.n_cross)
               & (df["hash_size"] == base.hash_size)]
    std = seeds["test_auc"].std(ddof=1) if len(seeds) > 1 else 0.0
    fig, axes = plt.subplots(1, len(knobs), figsize=(3.6 * len(knobs), 3.2), sharey=True)
    ink, muted, accent = "#1f2933", "#7b8794", "#2563eb"
    for ax, knob in zip(axes, knobs, strict=True):
        others = [k for k in GRID if k != knob]
        rows = df[(df["seed"] == SEED)]
        for k in others:
            rows = rows[rows[k] == getattr(base, k)]
        rows = rows.sort_values(knob)
        x = list(range(len(rows)))
        ax.plot(x, rows["test_auc"], color=accent, lw=2, marker="o", ms=8, zorder=3)
        for xi, (value, auc) in zip(x, zip(rows[knob], rows["test_auc"], strict=True),
                                    strict=True):
            is_base = value == getattr(base, knob)
            if is_base and std > 0:  # +/- 1 std over seeds: differences smaller are noise
                ax.errorbar(xi, auc, yerr=std, color=ink, capsize=4, lw=1, zorder=2)
            ax.annotate(f"{auc:.4f}", (xi, auc), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=ink if is_base else muted)
        ticks = [f"2^{int(v).bit_length() - 1}" if knob == "hash_size" else str(v)
                 for v in rows[knob]]
        ax.set_xticks(x, ticks)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xlabel(LABELS[knob], color=ink)
        ax.grid(axis="y", color="#e4e7eb", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("test AUC", color=ink)
    fig.suptitle(f"{model}: one knob at a time (others at default; bar = +/-1 std over "
                 f"{len(seeds)} seeds)", fontsize=10, color=ink)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "ablations.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="dcnv2")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        print(f"main-table rows stay in {METRICS_CSV.name}; ablation rows -> {ABLATIONS_CSV.name}")
        train_all(args.model)
    plot(summarize(args.model), args.model)


if __name__ == "__main__":
    main()
