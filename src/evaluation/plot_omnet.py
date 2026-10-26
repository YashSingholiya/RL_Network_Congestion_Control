"""Figures comparing the Python fluid simulator with the OMNeT++ packet-level simulator.
Run after evaluate_omnet.py:  python -m src.evaluation.plot_omnet [--pattern markov]"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.plots import LABEL

OUT = "results/omnet"                      # changed by --out
STYLE = {"python": dict(ls="-", lw=1.6, alpha=0.9), "omnet": dict(ls="--", lw=1.4, alpha=0.95)}
COLOR = {"python": "#1f77b4", "omnet": "#d62728"}
NAME = {"python": "Python fluid simulator", "omnet": "OMNeT++ packet-level simulator"}


def timeseries(pattern="markov", methods=None):
    methods = methods or [m for m in ("fixed", "aimd", "tabular_q", "ppo", "dqn")
                          if os.path.exists(f"{OUT}/traces/{pattern}_{m}_omnet.csv")]
    if not methods:
        raise SystemExit("no traces found - run evaluate_omnet.py first")
    fig, axes = plt.subplots(len(methods), 3, figsize=(13, 2.7 * len(methods)), squeeze=False)
    for i, m in enumerate(methods):
        for sim in ("python", "omnet"):
            d = pd.read_csv(f"{OUT}/traces/{pattern}_{m}_{sim}.csv")
            t = d.t * 0.05
            if sim == "python":
                axes[i, 0].plot(t, d.capacity, "k:", lw=1, label="capacity")
            axes[i, 0].plot(t, d.rate, color=COLOR[sim], label=NAME[sim], **STYLE[sim])
            axes[i, 1].plot(t, d.queue, color=COLOR[sim], **STYLE[sim])
            axes[i, 2].plot(t, d.latency * 1000, color=COLOR[sim], **STYLE[sim])
        axes[i, 0].set_ylabel(LABEL.get(m, m), fontweight="bold")
        for a in axes[i]:
            a.grid(alpha=.3)
    for j, tt in enumerate(["sending rate vs capacity (Mbps)", "queue (Mbit)", "latency (ms)"]):
        axes[0, j].set_title(tt, fontsize=10); axes[-1, j].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Same controller, same link ({pattern}, seed 10000): Python fluid model vs OMNeT++ packets", fontsize=11)
    fig.tight_layout(); os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/python_vs_omnet_timeseries_{pattern}.png", dpi=170); plt.close(fig)


def summary_bars():
    df = pd.read_csv(f"{OUT}/episodes.csv")
    g = df.groupby(["method", "simulator"])[["reward", "util", "p95_lat_ms", "loss"]].mean().unstack("simulator")
    order = [m for m in ("fixed", "aimd", "tabular_q", "approx_q", "dqn", "ppo") if m in g.index]
    titles = {"reward": "Episode return", "util": "Link utilisation", "p95_lat_ms": "p95 latency (ms)", "loss": "Loss rate"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    x = np.arange(len(order))
    for ax, k in zip(axes, titles):
        for s, off in (("python", -0.2), ("omnet", 0.2)):
            ax.bar(x + off, [g.loc[m, (k, s)] for m in order], 0.4, color=COLOR[s], label=NAME[s])
        ax.set_xticks(x); ax.set_xticklabels([LABEL.get(m, m) for m in order], rotation=40, ha="right", fontsize=8)
        ax.set_title(titles[k], fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/python_vs_omnet_summary.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--pattern", default="markov"); p.add_argument("--out", default="results/omnet"); a = p.parse_args()
    OUT = a.out
    timeseries(a.pattern); summary_bars(); print(f"figures written to {OUT}/figures")
