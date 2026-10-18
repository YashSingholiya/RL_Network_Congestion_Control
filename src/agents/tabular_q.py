"""Tabular Q-learning on a binned (compact) state -- Lec 20-23.

State = the 5 features of the latest MI, each discretised into a few bins
(4*3*4*3*3 = 432 states). Epsilon-greedy exploration with linear decay.
Run:  python -m src.agents.tabular_q --episodes 3000 --seed 0
"""
import argparse
import os
import numpy as np

from src.config import ACTIONS, N_FEATURES, EnvConfig, robust_train_cfg
from src.envs.congestion_env import CongestionEnv
from src.utils import write_monitor_csv

BINS = [np.array([0.05, 0.2, 0.5]),    # latency excess
        np.array([-0.02, 0.02]),       # latency gradient
        np.array([0.9, 1.1, 1.5]),     # sending ratio
        np.array([0.001, 0.02]),       # loss rate
        np.array([0.5, 0.9])]          # normalised throughput
SHAPE = tuple(len(b) + 1 for b in BINS)
N_STATES = int(np.prod(SHAPE))


def discretize(obs):
    f = obs[-N_FEATURES:]
    idx = tuple(int(np.digitize(f[i], BINS[i])) for i in range(N_FEATURES))
    return int(np.ravel_multi_index(idx, SHAPE))


class TabularQAgent:
    reset_options = {}

    def __init__(self, n_actions=len(ACTIONS), seed=0):
        self.Q = np.zeros((N_STATES, n_actions))
        self.rng = np.random.default_rng(seed)

    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        return int(np.argmax(self.Q[discretize(obs)]))

    def update(self, s, a, r, s2, alpha, gamma):
        target = r + gamma * self.Q[s2].max()                 # Bellman target
        self.Q[s, a] += alpha * (target - self.Q[s, a])       # TD update

    def save(self, path):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path):
        agent = cls()
        agent.Q = np.load(path)
        return agent


def train(episodes=3000, seed=0, robust=False, alpha=0.1, gamma=0.95, eps0=1.0, eps_min=0.05):
    tag = "_robust" if robust else ""
    env = CongestionEnv(robust_train_cfg() if robust else EnvConfig())
    agent = TabularQAgent(seed=seed)
    returns, lengths = [], []
    for ep in range(episodes):
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / (0.7 * episodes))
        obs, _ = env.reset(seed=seed * 100000 + ep)
        s, total, done = discretize(obs), 0.0, False
        steps = 0
        while not done:
            a = agent.act(obs, eps)
            obs, r, term, trunc, _ = env.step(a)
            s2 = discretize(obs)
            agent.update(s, a, r, s2, alpha, gamma)   # episodes only truncate -> always bootstrap
            s, total, steps, done = s2, total + r, steps + 1, term or trunc
        returns.append(total)
        lengths.append(steps)
        if (ep + 1) % 250 == 0:
            print(f"ep {ep+1}/{episodes}  eps={eps:.2f}  avg return(last 250)={np.mean(returns[-250:]):.1f}")
    os.makedirs("models", exist_ok=True)
    agent.save(f"models/tabular_q{tag}_s{seed}.npy")
    write_monitor_csv(f"results/logs/tabular_q{tag}_s{seed}/0.monitor.csv", returns, lengths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robust", action="store_true", help="randomise latency noise during training")
    a = p.parse_args()
    train(a.episodes, a.seed, a.robust)
