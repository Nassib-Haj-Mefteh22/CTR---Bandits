import numpy as np

from bandit.policies import LinUCB
from bandit.run import best_fixed_arm_regret, run_policy
from bandit.simulator import BanditEnv, CriteoBanditFactory

EVAL_ROUNDS = 20_000


def greedy_regret(scores: np.ndarray, env: BanditEnv) -> float:
    arm = scores.argmax(axis=1)
    return float((env.probs.max(axis=1) - env.probs[np.arange(len(arm)), arm]).sum())


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def main() -> None:
    factory = CriteoBanditFactory()
    test = factory.make_env(1, EVAL_ROUNDS)
    fixed = np.tile(test.probs.mean(axis=0), (EVAL_ROUNDS, 1))
    print(f"greedy regret on {EVAL_ROUNDS:,} seed-1 rounds")
    print(f"  best fixed arm                              {greedy_regret(fixed, test):6.0f}")
    train = factory.make_env(0, 20_000)
    w = ridge(train.features, train.probs)
    noise_free = greedy_regret(test.features @ w, test)
    print(f"  linear, full info, noise-free p             {noise_free:6.0f}")
    for n in (2_000, 20_000, 100_000, 200_000):
        train = factory.make_env(0, n)
        w = ridge(train.features, train.rewards)
        print(f"  linear, full info, {n:>7,} 0/1 rewards/arm   "
              f"{greedy_regret(test.features @ w, test):6.0f}")

    env = factory.make_env(1, 200_000)
    lin, fix = run_policy(env, LinUCB(env.n_arms, env.features.shape[1], 1, alpha=0.1)), \
        best_fixed_arm_regret(env)
    print("LinUCB (alpha 0.1) vs best fixed arm, seed 1")
    for t in (20_000, 50_000, 100_000, 200_000):
        q = t // 4
        lin_q, fix_q = (lin[t - 1] - lin[t - q - 1]) / q, (fix[t - 1] - fix[t - q - 1]) / q
        print(f"  T {t:>7,}: cumulative {lin[t - 1]:6.0f} vs {fix[t - 1]:6.0f}; per round in last "
              f"quarter {lin_q:.4f} vs {fix_q:.4f}")


if __name__ == "__main__":
    main()
