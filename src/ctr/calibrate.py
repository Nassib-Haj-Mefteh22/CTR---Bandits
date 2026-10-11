"""Calibration: does p = 0.3 mean "30% of these rows are clicked"? Measure it, then fix it.

ECE (expected calibration error): sort predictions into 15 equal-width bins of p, and in each bin
compare the mean prediction (confidence) with the observed click rate (accuracy); ECE is the
row-weighted mean gap. For binary CTR we bin p(click) itself, not max(p, 1-p) as in the
multi-class version of Guo et al. 2017, because the bidder consumes p(click).

Fixes, both fitted on VAL logits z only (test stays untouched), both monotonic in z, so the
ORDER of rows and therefore the AUC cannot change:
- temperature scaling: p = sigmoid(z / T), one parameter (Guo et al. 2017);
- Platt scaling (extra, for comparison): p = sigmoid(a z + b). The bias b can also correct a
  global shift, which T alone cannot (T only stretches logits around z = 0, i.e. p = 0.5).

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe -m ctr.calibrate --model dcnv2
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ctr.config import FIGURES_DIR, RESULTS_DIR, SEED  # noqa: E402
from ctr.data import CTRDataset, load_split, make_loader  # noqa: E402
from ctr.metrics import ctr_metrics  # noqa: E402
from ctr.train import CHECKPOINT_DIR, ModelConfig, build_model  # noqa: E402

N_BINS = 15
CALIBRATION_CSV = RESULTS_DIR / "calibration.csv"


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width bin of p: (mean prediction, observed click rate, row count).

    Empty bins get NaN means and count 0, so they weigh nothing in the ECE.
    """
    # bin index 0..n_bins-1; p = 1.0 goes into the last bin instead of an extra one
    idx = np.minimum((y_prob * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=y_prob, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y_true, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_p / count, sum_y / count, count


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """sum_b (n_b / N) |mean p in b - click rate in b|; 0 = perfectly calibrated."""
    conf, acc, count = reliability_bins(y_true, y_prob, n_bins)
    keep = count > 0
    return float(np.sum(count[keep] / count.sum() * np.abs(conf[keep] - acc[keep])))


def _fit(logits: np.ndarray, labels: np.ndarray, params: list[torch.Tensor], transform) -> None:
    """Minimize BCE of transform(z) over `params` with LBFGS (full batch: 300k scalars)."""
    z = torch.from_numpy(logits.astype(np.float64))
    y = torch.from_numpy(labels.astype(np.float64))
    optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(transform(z), y)
        loss.backward()
        return loss

    optimizer.step(closure)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T > 0 minimizing val LogLoss of sigmoid(z / T). Optimized as log T so T stays positive."""
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    _fit(logits, labels, [log_t], lambda z: z / log_t.exp())
    return float(log_t.detach().exp())


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(a, b) minimizing val LogLoss of sigmoid(a z + b); starts at the identity a=1, b=0."""
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    _fit(logits, labels, [a, b], lambda z: a * z + b)
    return float(a.detach()), float(b.detach())


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """(labels, raw logits): calibration maps act on logits, not on probabilities."""
    model.eval()
    labels, logits = [], []
    for dense, sparse, label in loader:
        logits.append(model(dense, sparse).double().numpy())
        labels.append(label.numpy())
    return np.concatenate(labels), np.concatenate(logits)


def load_trained(name: str, seed: int, mcfg: ModelConfig) -> nn.Module:
    model = build_model(name, mcfg)
    path = CHECKPOINT_DIR / f"{name}_{mcfg.tag()}_s{seed}.pt"
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def platt_params(model: str = "dcnv2", seed: int = SEED) -> tuple[float, float]:
    """(a, b) fitted on val by main(), read back from results/calibration.csv."""
    with CALIBRATION_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == model and row["seed"] == str(seed) and row["method"] == "platt":
                return float(row["a"]), float(row["b"])
    raise KeyError(f"no Platt row for {model} seed {seed}: run ctr.calibrate first")


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = ctr_metrics(y, p)
    return {"auc": m["auc"], "logloss": m["logloss"], "ece": ece(y, p), "mean_pred": m["mean_pred"]}


def calibrate_one(name: str, seed: int) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    """Fit T and (a, b) on val for one trained model; return csv rows + test probabilities."""
    model = load_trained(name, seed, ModelConfig())
    loaders = {s: make_loader(CTRDataset(*load_split(s)), 65536, shuffle=False)
               for s in ("val", "test")}
    y_val, z_val = predict_logits(model, loaders["val"])
    y_test, z_test = predict_logits(model, loaders["test"])
    t = fit_temperature(z_val, y_val)
    a, b = fit_platt(z_val, y_val)
    print(f"{name} seed {seed}: T = {t:.4f}   Platt a = {a:.4f}, b = {b:+.4f}")
    test_probs = {
        "uncalibrated": sigmoid(z_test),
        "temperature": sigmoid(z_test / t),
        "platt": sigmoid(a * z_test + b),
    }
    val_probs = {"uncalibrated": sigmoid(z_val), "temperature": sigmoid(z_val / t),
                 "platt": sigmoid(a * z_val + b)}
    rows = []
    for method, p in test_probs.items():
        val, test = evaluate(y_val, val_probs[method]), evaluate(y_test, p)
        rows.append({"model": name, "seed": seed, "method": method, "T": round(t, 4),
                     "a": round(a, 4), "b": round(b, 4),
                     **{f"val_{k}": round(v, 5) for k, v in val.items()},
                     **{f"test_{k}": round(v, 5) for k, v in test.items()},
                     "test_ctr": round(float(y_test.mean()), 4)})
        print(f"  {method:12s} test AUC {test['auc']:.5f}  LogLoss {test['logloss']:.5f}  "
              f"ECE {test['ece']:.5f}  mean_pred {test['mean_pred']:.4f} "
              f"(CTR {y_test.mean():.4f})")
    return rows, test_probs, y_test


def plot_reliability(y: np.ndarray, probs: dict[str, np.ndarray], title: str) -> None:
    """Reliability diagram, the gap (observed - predicted) per bin, and the histogram of p.

    Why a gap panel: on a 0-1 axis, errors of ~0.01 are invisible and every method sits on the
    diagonal; plotting the gap itself makes over/under-prediction readable.
    """
    style = {"uncalibrated": ("#7b8794", "o", "-"), "temperature": ("#2563eb", "s", "-"),
             "platt": ("#d97706", "^", "--")}
    ink, grid = "#1f2933", "#e4e7eb"
    fig, (ax, gap, hist) = plt.subplots(3, 1, figsize=(5.2, 8.0), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 1]})
    ax.plot([0, 1], [0, 1], color=ink, lw=1, ls=":", label="perfect calibration")
    gap.axhline(0, color=ink, lw=1, ls=":")
    for method, p in probs.items():
        color, marker, ls = style[method]
        conf, acc, count = reliability_bins(y, p)
        keep = count >= 100  # bins with < 100 rows are too noisy to read
        kw = {"color": color, "marker": marker, "ms": 6, "lw": 2, "ls": ls}
        ax.plot(conf[keep], acc[keep], label=f"{method} (ECE {ece(y, p):.4f})", **kw)
        gap.plot(conf[keep], acc[keep] - conf[keep], **kw)
    ax.set_ylabel("observed click rate", color=ink)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=10, color=ink)
    gap.set_ylabel("observed - predicted", color=ink)
    hist.hist(probs["uncalibrated"], bins=np.linspace(0, 1, N_BINS + 1), color="#7b8794",
              edgecolor="white", linewidth=2)
    hist.set_xlabel("predicted click probability (15 bins)", color=ink)
    hist.set_ylabel("test rows", color=ink)
    for a in (ax, gap, hist):
        a.grid(color=grid, lw=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "reliability.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="dcnv2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 1, SEED + 2])
    args = parser.parse_args()
    all_rows = []
    for i, seed in enumerate(args.seeds):
        rows, test_probs, y_test = calibrate_one(args.model, seed)
        all_rows += rows
        if i == 0:
            plot_reliability(y_test, test_probs, f"{args.model} seed {seed}, test split")
    with CALIBRATION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"rows -> {CALIBRATION_CSV}")
    for method in ("uncalibrated", "temperature", "platt"):
        r = [row for row in all_rows if row["method"] == method]
        for k in ("test_ece", "test_logloss", "test_auc"):
            v = np.array([row[k] for row in r])
            print(f"{method:12s} {k:13s} mean {v.mean():.5f}  std {v.std(ddof=1):.5f}  "
                  f"over seeds {args.seeds}")


if __name__ == "__main__":
    main()
