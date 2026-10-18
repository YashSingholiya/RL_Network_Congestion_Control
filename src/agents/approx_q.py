"""Approximate Q-learning with linear value functions -- Lec 24.

Q(s,a) = sum_i w_i * f_i(s,a). Here f(s,a) = one-hot(a) (x) phi(s), i.e. one
weight vector per action over the feature vector phi(s). Update (Lec 24):
    w_i <- w_i + alpha * [r + gamma*max_a' Q(s',a') - Q(s,a)] * f_i(s,a)
Run:  python -m src.agents.approx_q --episodes 3000 --seed 0
"""
import argparse
import os
import numpy as np

from src.config import ACTIONS, N_FEATURES, EnvConfig, robust_train_cfg
from src.envs.congestion_env import CongestionEnv
from src.utils import write_monitor_csv


def phi(obs):
    f = obs[-N_FEATURES:].astype(np.float64)
    return np.array([1.0, f[0] / 5.0, f[1], f[2] / 3.0, f[3], f[4]])


class ApproxQAgent:
    reset_options = {}

    def __init__(self, n_actions=len(ACTIONS), seed=0):
        self.W = np.zeros((n_actions, 6))
        self.rng = np.random.default_rng(seed)

    def q(self, obs):
        return self.W @ phi(obs)

    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.W.shape[0]))
        return int(np.argmax(self.q(obs)))

    def update(self, obs, a, r, obs2, alpha, gamma):
        diff = r + gamma * self.q(obs2).max() - self.q(obs)[a]
        diff = float(np.clip(diff, -5.0, 5.0))               # keep updates stable
        self.W[a] += alpha * diff * phi(obs)

    def save(self, path):
        np.save(path, self.W)

    @classmethod
    def load(cls, path):
        agent = cls()
        agent.W = np.load(path)
        return agent


def train(episodes=3000, seed=0, robust=False, alpha=0.01, gamma=0.95, eps0=1.0, eps_min=0.05):
    tag = "_robust" if robust else ""
    env = CongestionEnv(robust_train_cfg() if robust else EnvConfig())
    agent = ApproxQAgent(seed=seed)
    returns, lengths = [], []
    for ep in range(episodes):
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / (0.7 * episodes))
        obs, _ = env.reset(seed=seed * 100000 + ep)
        total, steps, done = 0.0, 0, False
        while not done:
            a = agent.act(obs, eps)
            obs2, r, term, trunc, _ = env.step(a)
            agent.update(obs, a, r, obs2, alpha, gamma)
            obs, total, steps, done = obs2, total + r, steps + 1, term or trunc
        returns.append(total)
        lengths.append(steps)
        if (ep + 1) % 250 == 0:
            print(f"ep {ep+1}/{episodes}  eps={eps:.2f}  avg return(last 250)={np.mean(returns[-250:]):.1f}")
    os.makedirs("models", exist_ok=True)
    agent.save(f"models/approx_q{tag}_s{seed}.npy")
    write_monitor_csv(f"results/logs/approx_q{tag}_s{seed}/0.monitor.csv", returns, lengths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robust", action="store_true", help="randomise latency noise during training")
    a = p.parse_args()
    train(a.episodes, a.seed, a.robust)
