"""Evaluate every method on every capacity scenario with identical episodes.

Run:  python -m src.evaluation.evaluate --episodes 20 --seeds 0
Outputs: results/eval_episodes.csv, results/summary.csv, results/traces/*.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

from src.config import SCENARIOS
from src.envs.congestion_env import CongestionEnv
from src.baselines.simple import RandomPolicy, FixedRatePolicy
from src.baselines.aimd import AIMDPolicy
from src.agents.tabular_q import TabularQAgent
from src.agents.approx_q import ApproxQAgent

METHODS = ["random", "fixed", "aimd", "tabular_q", "approx_q", "dqn", "ppo",
           "tabular_q_robust", "approx_q_robust", "dqn_robust", "ppo_robust"]
DETERMINISTIC = ("random", "fixed", "aimd")   # no trained model / seeds needed


class SB3Policy:
    reset_options = {}

    def __init__(self, algo, path):
        from stable_baselines3 import DQN, PPO
        self.model = (DQN if algo.startswith("dqn") else PPO).load(path, device="cpu")

    def act(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)


def load_policies(method, seeds):
    """Return a list of (agent_seed, policy) for a method; skips missing models."""
    if method == "random":
        return [(0, RandomPolicy(0))]
    if method == "fixed":
        return [(0, FixedRatePolicy(1.0))]
    if method == "aimd":
        return [(0, AIMDPolicy())]
    out = []
    for s in seeds:
        if method.startswith("tabular_q"):
            path, ctor = f"models/{method}_s{s}.npy", TabularQAgent.load
        elif method.startswith("approx_q"):
            path, ctor = f"models/{method}_s{s}.npy", ApproxQAgent.load
        else:
            path, ctor = f"models/{method}_s{s}.zip", (lambda p, m=method: SB3Policy(m, p))
        if os.path.exists(path):
            out.append((s, ctor(path)))
        else:
            print(f"[skip] {path} not found")
    return out


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed, options=policy.reset_options)
    rows, done = [], False
    while not done:
        a = policy.act(obs, env) if getattr(policy, "uses_env", False) else policy.act(obs)
        obs, r, term, trunc, info = env.step(a)
        rows.append({**info, "reward": r})
        done = term or trunc
    df = pd.DataFrame(rows)
    return dict(
        reward=df.reward.sum(),
        util=df.throughput.sum() / df.capacity.sum(),
        avg_lat_ms=1000 * df.latency.mean(),
        p95_lat_ms=1000 * np.percentile(df.latency, 95),
        loss=(df.overflow.sum() + df.rand_lost.sum()) / df.arrivals.sum(),
    ), df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--rand-loss", type=float, default=0.0)
    a = p.parse_args()

    from src.config import EnvConfig
    cfg = EnvConfig(rand_loss=a.rand_loss)
    os.makedirs("results/traces", exist_ok=True)
    records = []
    for sc in SCENARIOS:
        env = CongestionEnv(cfg, scenario=sc)
        for method in METHODS:
            for agent_seed, pol in load_policies(method, a.seeds):
                for ep in range(a.episodes):
                    m, trace = run_episode(env, pol, seed=10_000 + ep)   # same episodes for all methods
                    records.append(dict(method=method, scenario=sc, agent_seed=agent_seed, ep=ep, **m))
                    if ep == 0 and agent_seed == (a.seeds[0] if method not in DETERMINISTIC else 0):
                        trace[["capacity", "rate", "throughput", "latency", "queue"]].to_csv(
                            f"results/traces/{sc}_{method}.csv", index_label="t")
    df = pd.DataFrame(records)
    df.to_csv("results/eval_episodes.csv", index=False)

    cols = ["reward", "util", "avg_lat_ms", "p95_lat_ms", "loss"]
    per_sc = df.groupby(["method", "scenario"])[cols].agg(["mean", "std"])
    overall = df.groupby("method")[cols].agg(["mean", "std"])
    per_sc.to_csv("results/summary_by_scenario.csv")
    overall.to_csv("results/summary.csv")
    pd.set_option("display.width", 200)
    print("\n=== Overall (all scenarios) ===")
    print(df.groupby("method")[cols].mean().loc[[m for m in METHODS if m in df.method.unique()]].round(3).to_string())


if __name__ == "__main__":
    main()
