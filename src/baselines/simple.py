"""Non-learning reference policies (no networking knowledge needed)."""
import numpy as np
from src.config import ACTIONS, HOLD_IDX


class RandomPolicy:
    """Picks a random rate change every step -- the floor baseline."""
    reset_options = {}

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(len(ACTIONS)))


class FixedRatePolicy:
    """Never changes the rate. The rate is set once to `frac` x mean capacity."""

    def __init__(self, frac=1.0):
        self.reset_options = {"init_rate_frac": frac}

    def act(self, obs):
        return HOLD_IDX
