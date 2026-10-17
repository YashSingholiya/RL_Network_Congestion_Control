"""Small helpers shared by the training scripts."""
import os
import numpy as np


def write_monitor_csv(path, returns, lengths):
    """Write a Stable-Baselines3-compatible monitor file (r,l,t) so every
    algorithm's training curve can be plotted with the same code."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.cumsum(np.ones(len(returns)))
    with open(path, "w") as f:
        f.write("#{}\n")
        f.write("r,l,t\n")
        for r, l, ti in zip(returns, lengths, t):
            f.write(f"{r:.4f},{int(l)},{ti:.1f}\n")
