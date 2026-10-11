"""Run every policy on the Criteo bandit: tune on seed 0, report on seeds 1-5.

Why a separate tuning seed: alpha (LinUCB) and eps (eps-greedy) are hyperparameters; picking them
on the seeds we report would make the reported regret optimistic, the bandit version of tuning on
test. Seed 0 gets its own contexts and coin flips, so the choice transfers but does not leak.

Outputs: results/bandit_tuning.csv (seed 0 sweep), results/bandit_regret.csv (cumulative regret
every 1000 rounds per policy and seed, incl. the best-fixed-arm reference).
Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe -m bandit.run
"""

import argparse
import csv
import time
from collections.abc import Callable

import numpy as np

from bandit.policies import EpsilonGreedy, LinearTS, LinUCB, NeuralLinear, RandomPolicy
from bandit.simulator import BanditEnv, CriteoBanditFactory
from ctr.config import RESULTS_DIR

# 20k was too short: with ~2k noisy pulls per arm even a full-information linear fit loses to the
# best fixed arm (Step 6 diagnostic); LinUCB overtakes it in total around 100k rounds.
HORIZON = 100_000
TUNE_SEED = 0
EVAL_SEEDS = [1, 2, 3, 4, 5]
LOG_EVERY = 1000
# name -> (factory(n_arms, dim, seed, **param), hyperparameter name, grid swept on TUNE_SEED)
POLICIES: dict[str, tuple[Callable, str | None, list[float]]] = {
    "random": (lambda k, d, s: RandomPolicy(k, s), None, []),
    "eps-greedy": (lambda k, d, s, eps: EpsilonGreedy(k, s, eps=eps), "eps",
                   [0.02, 0.05, 0.1, 0.2]),
    "linucb": (lambda k, d, s, alpha: LinUCB(k, d, s, alpha=alpha), "alpha",
               [0.01, 0.02, 0.05, 0.1, 0.25]),
    # Thompson policies: no exploration knob to tune (the posterior sets the exploration);
    # prior and network settings are fixed in policies.py, not chosen on these runs.
    "linear-ts": (lambda k, d, s: LinearTS(k, d, s), None, []),
    "neural-linear": (lambda k, d, s: NeuralLinear(k, d, s), None, []),
}


def run_policy(env: BanditEnv, policy) -> np.ndarray:
    """Play all T rounds; returns cumulative pseudo-regret [T]."""
    regret = np.empty(env.horizon)
    for t in range(env.horizon):
        x = env.context(t)
        arm = policy.select(x)
        reward, regret[t] = env.pull(t, arm)
        policy.update(x, arm, reward)
    return np.cumsum(regret)


def best_fixed_arm_regret(env: BanditEnv) -> np.ndarray:
    """Reference: always play the arm with the best average p in hindsight (non-contextual
    optimum). A context-free policy cannot beat it; a contextual one must, to justify itself."""
    best = env.probs.mean(axis=0).argmax()
    return np.cumsum(env.probs.max(axis=1) - env.probs[:, best])


def make(name: str, env: BanditEnv, seed: int, value: float | None):
    factory, param, _ = POLICIES[name]
    k, d = env.n_arms, env.features.shape[1]
    return factory(k, d, seed) if param is None else factory(k, d, seed, value)


def tune(factory: CriteoBanditFactory, horizon: int) -> dict[str, float | None]:
    env = factory.make_env(TUNE_SEED, horizon)
    chosen, rows = {}, []
    for name, (_, param, grid) in POLICIES.items():
        if param is None:
            chosen[name] = None
            continue
        finals = {v: run_policy(env, make(name, env, TUNE_SEED, v))[-1] for v in grid}
        chosen[name] = min(finals, key=finals.get)
        for v, r in finals.items():
            rows.append({"policy": name, "param": param, "value": v, "final_regret": round(r, 1),
                         "chosen": v == chosen[name]})
        print(f"tune {name:10s} {param}: " + "  ".join(f"{v}->{r:.0f}" for v, r in finals.items())
              + f"  => {chosen[name]}")
    write_csv(RESULTS_DIR / "bandit_tuning.csv", rows)
    return chosen


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    start = time.perf_counter()
    factory = CriteoBanditFactory()
    print(f"arms (C6 ids) {factory.arms.tolist()}, coverage {factory.coverage:.3f}, "
          f"setup {time.perf_counter() - start:.0f}s")
    chosen = tune(factory, args.horizon)

    rows, finals, full = [], {}, {}
    steps = list(range(LOG_EVERY, args.horizon + 1, LOG_EVERY))
    for seed in EVAL_SEEDS:
        env = factory.make_env(seed, args.horizon)
        curves = {"best-fixed-arm": best_fixed_arm_regret(env)}
        for name in POLICIES:
            t0 = time.perf_counter()
            curves[name] = run_policy(env, make(name, env, seed, chosen[name]))
            print(f"seed {seed} {name:10s} final regret {curves[name][-1]:7.1f}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
        for name, curve in curves.items():
            finals.setdefault(name, []).append(curve[-1])
            full.setdefault(name, []).append(curve)
            rows += [{"policy": name, "seed": seed, "t": t, "cum_regret": round(curve[t - 1], 2)}
                     for t in steps]
    write_csv(RESULTS_DIR / "bandit_regret.csv", rows)

    print(f"\nfinal cumulative regret at T = {args.horizon}, seeds {EVAL_SEEDS} (mean +/- std):")
    for name, v in finals.items():
        v = np.array(v)
        print(f"  {name:15s} {v.mean():7.1f} +/- {v.std(ddof=1):5.1f}   per round "
              f"{v.mean() / args.horizon:.4f}")
    # linear growth check: regret per round in the last quarter vs the first quarter, from the
    # full curves (the csv only keeps every LOG_EVERY-th round)
    q = args.horizon // 4
    for name in ("random", "linucb", "linear-ts", "neural-linear"):
        c = np.array(full[name])
        first = c[:, q - 1].mean() / q
        last = (c[:, -1] - c[:, -q - 1]).mean() / q
        print(f"  {name}: regret/round first quarter {first:.4f}, last quarter {last:.4f}")


if __name__ == "__main__":
    main()
