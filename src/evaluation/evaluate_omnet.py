"""Run the same controllers on the SAME episodes in both simulators (Python fluid model and OMNeT++ packets).

Run (after building omnetpp/, see omnetpp/README.md):
    python -m src.evaluation.evaluate_omnet --episodes 3 --seeds 0
Outputs (--out, default results/omnet/):
    episodes.csv, summary.csv, traces/<pattern>_<method>_<simulator>.csv
Then:  python -m src.evaluation.plot_omnet   for the comparison figures.
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.config import EnvConfig, SCENARIOS
from src.envs.congestion_env import CongestionEnv
from src.envs.omnet_env import OmnetCongestionEnv, omnet_available, default_exe
from src.evaluation.evaluate import load_policies, run_episode

DEFAULT_METHODS = ["fixed", "aimd", "tabular_q", "approx_q", "dqn", "ppo"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3, help="episodes per (pattern, method); seeds 10000, 10001, ...")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="trained-agent seeds to load")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--patterns", nargs="+", default=SCENARIOS)
    p.add_argument("--py-noise", type=float, default=0.02, help="latency noise of the Python simulator (training default 0.02)")
    p.add_argument("--out", default="results/omnet", help="output folder")
    p.add_argument("--latency", choices=["last", "mean", "queue"], default="last", help="which RTT statistic OMNeT++ reports")
    a = p.parse_args()

    OUT = a.out
    if not omnet_available():
        raise SystemExit(f"OMNeT++ model not found at {default_exe()}. Build it first (omnetpp/README.md) "
                         "or set OMNETPP_CONGESTION_EXE.")
    os.makedirs(f"{OUT}/traces", exist_ok=True)
    policies = {}
    for m in a.methods:
        try:
            policies[m] = load_policies(m, a.seeds)
        except Exception as e:                      # e.g. PyTorch / stable-baselines3 not installed for DQN / PPO
            print(f"[skip] {m}: {e}")
    cfg = EnvConfig(lat_noise=a.py_noise)
    records = []
    for sc in a.patterns:
        sims = {"python": CongestionEnv(cfg, scenario=sc),
                "omnet": OmnetCongestionEnv(EnvConfig(lat_noise=0.0), scenario=sc, latency=a.latency)}
        for method, plist in policies.items():
            for agent_seed, pol in plist:
                for ep in range(a.episodes):
                    for sim_name, env in sims.items():
                        m, trace = run_episode(env, pol, seed=10_000 + ep)         # identical link in both simulators
                        records.append(dict(simulator=sim_name, scenario=sc, method=method, agent_seed=agent_seed, ep=ep, **m))
                        if ep == 0 and agent_seed == a.seeds[0]:
                            trace[["capacity", "rate", "throughput", "latency", "queue", "loss", "reward"]].to_csv(
                                f"{OUT}/traces/{sc}_{method}_{sim_name}.csv", index_label="t")
        print(f"finished pattern: {sc}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(f"{OUT}/episodes.csv", index=False)
    cols = ["reward", "util", "avg_lat_ms", "p95_lat_ms", "loss"]
    summ = df.groupby(["method", "simulator"])[cols].mean().unstack("simulator")
    summ.to_csv(f"{OUT}/summary.csv")
    pd.set_option("display.width", 220)
    order = [m for m in a.methods if m in df.method.unique()]
    for c in cols:
        t = summ[c].loc[order]
        t["omnet - python"] = t["omnet"] - t["python"]
        print(f"\n=== {c} ===\n{t.round(3).to_string()}")


if __name__ == "__main__":
    main()
