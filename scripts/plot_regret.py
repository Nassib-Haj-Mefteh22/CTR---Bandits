import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ctr.config import FIGURES_DIR, RESULTS_DIR

STYLE = {
    "random": ("#7b8794", "-", "random"),
    "best-fixed-arm": ("#7b8794", "--", "best fixed arm (hindsight)"),
    "eps-greedy": ("#2a78d6", "-", "eps-greedy"),
    "linucb": ("#eb6834", "-", "LinUCB"),
    "linear-ts": ("#1baf7a", "-", "Linear TS"),
    "neural-linear": ("#eda100", "-", "NeuralLinear TS"),
}


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "bandit_regret.csv")
    stats = df.groupby(["policy", "t"])["cum_regret"].agg(["mean", "std"]).reset_index()
    horizon = df["t"].max()
    final = stats[stats["t"] == horizon].set_index("policy")
    final["per_round"] = final["mean"] / horizon
    final = final.sort_values("mean", ascending=False)
    print(f"final cumulative regret at T = {horizon:,} over {df['seed'].nunique()} seeds")
    print(final.round(4).to_string())
    final.round(4).to_csv(RESULTS_DIR / "bandit_final.csv")

    ink, muted, grid = "#1f2933", "#52606d", "#e4e7eb"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ends: list[tuple[float, str]] = []
    for policy in final.index:
        color, ls, label = STYLE[policy]
        s = stats[stats["policy"] == policy]
        ax.fill_between(s["t"], s["mean"] - s["std"], s["mean"] + s["std"], color=color,
                        alpha=0.18, lw=0)
        ax.plot(s["t"], s["mean"], color=color, ls=ls, lw=2, label=label)
        ends.append((s["mean"].iloc[-1], f"{label}  {s['mean'].iloc[-1]:,.0f}"))
    min_gap = 0.035 * stats["mean"].max()
    placed: list[float] = []
    for y, text in sorted(ends):
        y_text = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (horizon, y), xytext=(horizon * 1.01, y_text), va="center",
                    fontsize=8, color=ink)
    ax.set_xlabel("round t", color=ink)
    ax.set_ylabel("cumulative pseudo-regret", color=ink)
    ax.set_title("Criteo bandit (10 arms of C6): mean +/- 1 std over 5 seeds", fontsize=10,
                 color=ink)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.grid(color=grid, lw=0.8)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, horizon * 1.32)
    ax.set_xticks(range(0, horizon + 1, horizon // 5))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "regret.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
