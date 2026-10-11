"""Step 7 diagnostic: why NeuralLinear loses to linear policies on the Criteo bandit.

1. Variance split of the true p on seed-0 rounds: user part (mean over arms, the same for every
   ad), arm part (deviation of each arm from that mean), and Bernoulli noise p(1-p).
2. Representation quality: after a full 100k-round run on seed 0, play greedily on the posterior
   MEAN (no exploration) over 20k seed-1 rounds, with NeuralLinear's learned phi vs LinearTS's raw
   features. Same Bayesian layer, only the features differ.
Seed 0 is the tuning seed: the reported seeds 1-5 of bandit.run are never used here.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe scripts/neural_linear_diagnostic.py
"""

import numpy as np

from bandit.policies import LinearTS, NeuralLinear
from bandit.run import HORIZON, run_policy
from bandit.simulator import BanditEnv, CriteoBanditFactory


def greedy_regret(scores: np.ndarray, env: BanditEnv) -> float:
    arm = scores.argmax(axis=1)
    return float((env.probs.max(axis=1) - env.probs[np.arange(len(arm)), arm]).sum())


def main() -> None:
    factory = CriteoBanditFactory()
    env = factory.make_env(0, HORIZON)
    test = factory.make_env(1, 20_000)

    p = env.probs
    user = p.mean(axis=1, keepdims=True)
    v_user, v_arm, v_noise = user.var(), ((p - user) ** 2).mean(), (p * (1 - p)).mean()
    print("variance split of the reward, seed 0")
    print(f"  user (same for every arm)  {v_user:.5f}")
    share = v_arm / (v_user + v_arm)
    print(f"  arm (decides the choice)   {v_arm:.5f}  = {share:.1%} of the signal")
    print(f"  Bernoulli noise            {v_noise:.5f}")

    k, d = env.n_arms, env.features.shape[1]
    neural, linear = NeuralLinear(k, d, 0), LinearTS(k, d, 0)
    print(f"after {HORIZON:,} rounds on seed 0 (cumulative regret), then greedy on the posterior "
          f"mean over {test.horizon:,} seed-1 rounds")
    for name, pol, feats in (("neural-linear", neural, neural.features),
                             ("linear-ts", linear, lambda x: x)):
        regret = run_policy(env, pol)[-1]
        greedy = greedy_regret(feats(test.features) @ pol.bayes.mu.T, test)
        print(f"  {name:14s} run regret {regret:6.0f}   greedy-on-mean regret {greedy:5.0f}")
    fixed = np.tile(test.probs.mean(axis=0), (test.horizon, 1))
    print(f"  reference: best fixed arm on the same 20k rounds {greedy_regret(fixed, test):.0f}")


if __name__ == "__main__":
    main()
