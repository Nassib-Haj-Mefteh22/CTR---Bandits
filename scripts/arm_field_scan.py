"""Step 6.1 evidence: which categorical field makes a useful bandit "ad" field?

For each of the 26 fields, arms = its K most frequent train values. For N val contexts, set the
field to each arm and ask the calibrated DCN-v2 for p(click | context, arm). A good testbed needs:
- spread: arms differ (max - min p), otherwise every policy has ~0 regret;
- context value: mean_x max_a p(x, a) - max_a mean_x p(x, a) = per-round regret of the best FIXED
  arm against the contextual oracle. If ~0, a non-contextual bandit is already optimal and LinUCB /
  NeuralLinear have nothing to learn;
- coverage: share of train rows whose value is one of the K arms (low = the model is asked about
  combinations it rarely saw, i.e. extrapolation).

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe scripts/arm_field_scan.py
"""

import argparse
import csv

import numpy as np
import torch

from bandit.simulator import arm_probs, top_values
from ctr.calibrate import load_trained, platt_params
from ctr.config import PROCESSED_DIR, RESULTS_DIR, SEED, SPARSE_FIELDS
from ctr.data import load_split
from ctr.train import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    model = load_trained("dcnv2", SEED, ModelConfig()).eval()
    a, b = platt_params()
    dense_v, sparse_v, _ = load_split("val")
    rows = np.random.default_rng(SEED).choice(len(dense_v), args.n, replace=False)
    dense, sparse = torch.from_numpy(dense_v[rows]), torch.from_numpy(sparse_v[rows])
    train_sparse = np.load(PROCESSED_DIR / "train_sparse.npy", mmap_mode="r")

    results = []
    for f, name in enumerate(SPARSE_FIELDS):
        arms, coverage = top_values(np.asarray(train_sparse[:, f]), args.k)
        if len(arms) < args.k:
            print(f"{name}: only {len(arms)} real values, skipped")
            continue
        p = arm_probs(model, a, b, dense, sparse, f, arms)
        best = p.argmax(axis=1)
        top2 = np.sort(p, axis=1)[:, -2:]
        results.append({
            "field": name,
            "coverage": coverage,
            "arm_mean_p_min": p.mean(axis=0).min(),
            "arm_mean_p_max": p.mean(axis=0).max(),
            "spread": float((p.max(axis=1) - p.min(axis=1)).mean()),
            "gap_top2": float((top2[:, 1] - top2[:, 0]).mean()),
            "context_value": float(p.max(axis=1).mean() - p.mean(axis=0).max()),
            "n_best_arms": int(np.unique(best).size),
            "share_not_global_best": float((best != p.mean(axis=0).argmax()).mean()),
        })
    results.sort(key=lambda r: -r["context_value"])
    print(f"K = {args.k} arms per field, {args.n} val contexts, Platt DCN-v2 seed {SEED}")
    print(f"{'field':5s} {'cover':>6s} {'arm p range':>13s} {'spread':>7s} {'gap':>7s} "
          f"{'ctx val':>8s} {'#best':>5s} {'!global':>7s}")
    for r in results:
        print(f"{r['field']:5s} {r['coverage']:6.3f} {r['arm_mean_p_min']:6.3f}-"
              f"{r['arm_mean_p_max']:.3f} {r['spread']:7.4f} {r['gap_top2']:7.4f} "
              f"{r['context_value']:8.4f} {r['n_best_arms']:5d} {r['share_not_global_best']:7.3f}")
    out = RESULTS_DIR / "arm_field_scan.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
