"""Pick AIMD's additive step on held-out tuning episodes (seeds distinct from the evaluation
seeds), so the baseline is tuned fairly. Run: python -m scripts.tune_aimd"""
import numpy as np
from src.config import SCENARIOS, EnvConfig
from src.envs.congestion_env import CongestionEnv
from src.baselines.aimd import AIMDPolicy
from src.evaluation.evaluate import run_episode

if __name__ == "__main__":
    results = {}
    for beta in [0.5, 0.8]:                       # must be an available action multiplier
        for inc in [0.01, 0.02, 0.05, 0.1, 0.25]:
            rewards = []
            for sc in SCENARIOS:
                env = CongestionEnv(EnvConfig(), scenario=sc)
                for ep in range(10):
                    m, _ = run_episode(env, AIMDPolicy(inc=inc, beta=beta), seed=50_000 + ep)
                    rewards.append(m["reward"])
            results[(inc, beta)] = float(np.mean(rewards))
            print(f"inc={inc:<5} beta={beta}  mean tuning reward = {results[(inc, beta)]:.1f}")
    best = max(results, key=results.get)
    print("best (inc, beta):", best)
