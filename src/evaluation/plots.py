"""Generate all report figures into results/figures/.  Run: python -m src.evaluation.plots"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "results/figures"
ORDER = ["random", "fixed", "aimd", "tabular_q", "approx_q", "dqn", "ppo",
         "tabular_q_robust", "approx_q_robust", "dqn_robust", "ppo_robust"]
LABEL = {"random": "Random", "fixed": "Fixed rate", "aimd": "AIMD", "tabular_q": "Tabular Q",
         "approx_q": "Approx. Q (linear)", "dqn": "DQN", "ppo": "PPO",
         "tabular_q_robust": "Tabular Q (robust)", "approx_q_robust": "Approx. Q (robust)",
         "dqn_robust": "DQN (robust)", "ppo_robust": "PPO (robust)"}
METRICS = ["reward", "util", "avg_lat_ms", "p95_lat_ms", "loss"]


# --------------------------------------------------------------------------------------
# Training curves: per-seed curve first, then mean +/- std ACROSS SEEDS
# --------------------------------------------------------------------------------------
def _seed_runs(method):
    """One (steps, returns) pair per training seed. Parallel envs of the same seed are
    merged into a single stream; step counts are total environment steps."""
    runs = []
    for d in sorted(glob.glob(f"results/logs/{method}_s*")):
        files = sorted(glob.glob(f"{d}/*.monitor.csv"))
        if not files:
            continue
        n = len(files)                       # n envs advance in lockstep -> total steps = n * per-env steps
        xs, rs = [], []
        for f in files:
            df = pd.read_csv(f, skiprows=1)
            xs.append(np.cumsum(df.l.values) * n)
            rs.append(df.r.values)
        runs.append((np.concatenate(xs), np.concatenate(rs)))
    return runs


def _binned(x, r, edges):
    """Mean episode return inside each step-bin; empty bins are filled by interpolation."""
    idx = np.digitize(x, edges) - 1
    m = np.full(len(edges) - 1, np.nan)
    for b in range(len(m)):
        sel = idx == b
        if sel.any():
            m[b] = r[sel].mean()
    ok = ~np.isnan(m)
    if ok.sum() == 0:
        return None
    return np.interp(np.arange(len(m)), np.where(ok)[0], m[ok])


def aggregate_curve(method, n_bins=40):
    """Returns (x, mean, std, per_seed_curves) over seeds, or None if no logs exist."""
    runs = _seed_runs(method)
    if not runs:
        return None
    x_max = min(x.max() for x, _ in runs)                 # range covered by every seed
    edges = np.linspace(0, x_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    curves = [c for c in (_binned(x, r, edges) for x, r in runs) if c is not None]
    M = np.vstack(curves)                                  # shape (n_seeds, n_bins)
    std = M.std(axis=0, ddof=1) if len(M) > 1 else np.zeros(M.shape[1])
    return centers, M.mean(axis=0), std, M


def training_curves(n_bins=40):
    panels = [("Deep RL", ["dqn", "ppo"]), ("Tabular / linear Q-learning", ["tabular_q", "approx_q"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (title, methods) in zip(axes, panels):
        for m in methods:
            agg = aggregate_curve(m, n_bins)
            if agg is None:
                continue
            x, mean, std, M = agg
            line, = ax.plot(x, mean, lw=1.8, label=f"{LABEL[m]} (n={len(M)} seeds)")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
            for c in M:                                    # thin per-seed curves
                ax.plot(x, c, lw=0.6, alpha=0.3, color=line.get_color())
        ax.set_title(title, fontsize=10); ax.set_xlabel("environment steps")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("episode return (training)")
    fig.suptitle("Training curves: mean +/- 1 std across seeds", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/training_curves.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------
# Evaluation figures
# --------------------------------------------------------------------------------------
def time_series(scenario="markov", methods=("fixed", "aimd", "dqn", "ppo")):
    """Rate vs capacity (top) and latency (bottom) for one episode. Few methods + legend
    above the axes keep the figure readable; pass methods=ORDER for the full version."""
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True)
    cap_done = False
    for m in methods:
        f = f"results/traces/{scenario}_{m}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if not cap_done:
            axes[0].plot(d.t * 0.05, d.capacity, "k--", lw=1.6, label="Capacity"); cap_done = True
        axes[0].plot(d.t * 0.05, d.rate, label=LABEL[m], lw=1.1)
        axes[1].plot(d.t * 0.05, d.latency * 1000, label=LABEL[m], lw=1.1)
    axes[0].set_ylabel("rate (Mbps)"); axes[1].set_ylabel("latency (ms)"); axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    for a in axes: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/time_series_{scenario}.png", dpi=200); plt.close(fig)


def _units(df):
    """One value per (method, agent_seed, episode): the mean over the capacity scenarios.
    Averaging over scenarios first removes the large between-scenario spread (e.g. reward
    400 on 'constant' vs ~40 on 'markov' for a fixed-rate sender) from the error bars."""
    return df.groupby(["method", "agent_seed", "ep"])[METRICS].mean().reset_index()


def bars():
    u = _units(pd.read_csv("results/eval_episodes.csv"))
    methods = [m for m in ORDER if m in u.method.unique()]
    titles = {"reward": "Episode return", "util": "Link utilisation",
              "p95_lat_ms": "p95 latency (ms)", "loss": "Loss rate"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    for ax, k in zip(axes, titles):
        vals = [u[u.method == m][k] for m in methods]
        ax.bar(range(len(methods)), [v.mean() for v in vals],
               yerr=[1.96 * v.std(ddof=1) / np.sqrt(len(v)) for v in vals], capsize=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABEL[m] for m in methods], rotation=45, ha="right", fontsize=7)
        ax.set_title(titles[k], fontsize=9)
    fig.suptitle("Mean over scenarios, 95% CI over episodes", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/bars.png", dpi=200); plt.close(fig)


def pareto():
    df = pd.read_csv("results/eval_episodes.csv")
    g = df.groupby(["method", "scenario"])[["util", "p95_lat_ms"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for m in ORDER:
        s = g[g.method == m]
        if len(s):
            ax.scatter(s.p95_lat_ms, s.util, label=LABEL[m], s=22)
    ax.set_xlabel("p95 latency (ms)"); ax.set_ylabel("utilisation"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("Throughput-delay trade-off (top-left is best)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/pareto.png", dpi=200); plt.close(fig)


def paired_report(pairs=(("ppo", "dqn"), ("dqn", "tabular_q"), ("ppo", "aimd"))):
    """Paired differences on identical episodes (same scenario + episode seed for all methods).
    Prints mean difference +/- 95% CI; if the interval contains 0 the difference is not clear.
    Note: this captures episode variability only, not training-seed variability."""
    df = pd.read_csv("results/eval_episodes.csv")
    print("\nPaired differences (A - B), mean +/- 95% CI over scenario x episode units")
    for a, b in pairs:
        if a not in df.method.values or b not in df.method.values:
            continue
        row = []
        for k in ["reward", "util", "p95_lat_ms", "loss"]:
            x = df[df.method == a].groupby(["scenario", "ep"])[k].mean()   # average over agent seeds
            y = df[df.method == b].groupby(["scenario", "ep"])[k].mean()
            d = (x - y).dropna()
            row.append(f"{k}: {d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")
        print(f"  {LABEL[a]} - {LABEL[b]}:  " + " | ".join(row))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    training_curves(); bars(); pareto()
    for sc in ["markov", "step", "trace"]:
        time_series(sc)
    paired_report()
    print("figures written to", FIG)