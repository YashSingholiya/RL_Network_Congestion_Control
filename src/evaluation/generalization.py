"""Zero-shot generalisation tests (no retraining): held-out capacity patterns and
congestion levels outside the training ranges.

Training distribution: 5 patterns (constant, step, sine, markov, trace); RTT 40-150 ms;
trough 20-60% of peak; buffer 1-4 BDP; initial rate 0.1-1.5x capacity; 2% latency noise;
no random loss. (Capacity *scale* is not tested: the fluid simulator is scale-free, so the
normalised features and rewards behave identically at any Mbps level.)

Run:  python -m src.evaluation.generalization --episodes 10 --seeds 0 1 2
Outputs: results/generalization_episodes.csv, results/generalization_summary.csv,
         results/figures/generalization.png
"""
import argparse
import dataclasses
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SCENARIOS, HELDOUT_SCENARIOS, EnvConfig
from src.envs.congestion_env import CongestionEnv
from src.evaluation.evaluate import METHODS, load_policies, run_episode
from src.evaluation.plots import LABEL

ALL_PATTERNS = SCENARIOS + HELDOUT_SCENARIOS

# suite name -> (capacity patterns, EnvConfig overrides)
SUITES = {
    "in_dist":          (SCENARIOS, {}),                                          # reference
    "heldout_patterns": (HELDOUT_SCENARIOS, {}),                                  # unseen patterns
    "shallow_buffer":   (SCENARIOS, dict(buffer_bdp_range=(0.25, 0.5))),          # trained 1-4 BDP
    "deep_buffer":      (SCENARIOS, dict(buffer_bdp_range=(6.0, 10.0))),
    "short_rtt":        (SCENARIOS, dict(rtt_range=(0.02, 0.03))),                # trained 40-150 ms
    "long_rtt":         (SCENARIOS, dict(rtt_range=(0.25, 0.40))),
    "overload_start":   (SCENARIOS, dict(init_rate_range=(2.0, 3.0))),            # trained 0.1-1.5x capacity
    "noisy_latency":    (SCENARIOS, dict(lat_noise=0.10)),                        # trained 2% noise
    "noisy_latency_20": (SCENARIOS, dict(lat_noise=0.20)),   # far beyond robust-training range (<=8%)
    "deep_fades":       (SCENARIOS, dict(cap_low_frac_range=(0.03, 0.12))),       # trained 0.2-0.6
    "rand_loss_1pct":   (SCENARIOS, dict(rand_loss=0.01)),
    "hard_combo":       (HELDOUT_SCENARIOS, dict(buffer_bdp_range=(0.25, 0.5),
                                                 cap_low_frac_range=(0.03, 0.12))),
}


def heatmap(ax, table, title, fmt, cmap):
    im = ax.imshow(table.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([LABEL[m] for m in table.index], fontsize=8)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, fmt.format(table.values[i, j]), ha="center", va="center", fontsize=6.5)
    ax.set_title(title, fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10, help="episodes per (suite, pattern, method)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--suites", nargs="+", default=list(SUITES))
    a = p.parse_args()

    policies = {m: load_policies(m, a.seeds) for m in METHODS}      # load every model once
    records = []
    for suite in a.suites:
        patterns, overrides = SUITES[suite]
        cfg = dataclasses.replace(EnvConfig(), **overrides)
        for sc in patterns:
            env = CongestionEnv(cfg, scenario=sc)
            for method, plist in policies.items():
                for agent_seed, pol in plist:
                    for ep in range(a.episodes):
                        m, _ = run_episode(env, pol, seed=20_000 + ep)   # same episodes for all methods
                        records.append(dict(suite=suite, scenario=sc, method=method,
                                            agent_seed=agent_seed, ep=ep, **m))
        print(f"finished suite: {suite}")

    df = pd.DataFrame(records)
    os.makedirs("results/figures", exist_ok=True)
    df.to_csv("results/generalization_episodes.csv", index=False)
    methods = [m for m in METHODS if m in df.method.unique()]
    suites = [s for s in SUITES if s in df.suite.unique()]

    def pivot(col):
        return df.groupby(["method", "suite"])[col].mean().unstack("suite").reindex(index=methods, columns=suites)

    reward, util, p95 = pivot("reward"), pivot("util"), pivot("p95_lat_ms")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n=== Mean episode return ===\n", reward.round(0).to_string())
    print("\n=== Link utilisation ===\n", util.round(2).to_string())
    print("\n=== p95 latency (ms) ===\n", p95.round(0).to_string())
    pd.concat({"reward": reward, "util": util, "p95_lat_ms": p95}, axis=1).to_csv("results/generalization_summary.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    heatmap(axes[0], reward, "Mean episode return (higher is better)", "{:.0f}", "RdYlGn")
    heatmap(axes[1], p95, "p95 latency, ms (lower is better)", "{:.0f}", "RdYlGn_r")
    fig.tight_layout(); fig.savefig("results/figures/generalization.png", dpi=200); plt.close(fig)
    print("saved results/figures/generalization.png")


if __name__ == "__main__":
    main()