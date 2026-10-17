"""Generators for time-varying bottleneck link capacity (Mbps), one value per step."""
import numpy as np


def make_capacity(kind, n, dt, low, high, rng):
    """Return an array of length n with the link capacity at every step.

    kind: constant | step | sine | markov | trace
    low/high: capacity range in Mbps; rng: numpy Generator.
    """
    if kind == "constant":
        cap = np.full(n, high)

    elif kind == "step":
        # 3-5 segments with random levels, abrupt changes in between
        n_seg = int(rng.integers(3, 6))
        cuts = np.sort(rng.choice(np.arange(20, n - 20), n_seg - 1, replace=False))
        levels = rng.uniform(low, high, n_seg)
        cap = np.empty(n)
        prev = 0
        for c, lv in zip(list(cuts) + [n], levels):
            cap[prev:c] = lv
            prev = c

    elif kind == "sine":
        t = np.arange(n) * dt
        period = rng.uniform(4.0, 10.0)
        phase = rng.uniform(0, 2 * np.pi)
        cap = (low + high) / 2 + (high - low) / 2 * np.sin(2 * np.pi * t / period + phase)

    elif kind == "markov":
        # random jumps between 3 capacity levels (low / mid / high)
        levels = np.linspace(low, high, 3)
        state = int(rng.integers(3))
        cap = np.empty(n)
        for i in range(n):
            if rng.random() < 0.03:
                state = int(rng.integers(3))
            cap[i] = levels[state]

    elif kind == "trace":
        # cellular-like: mean-reverting random walk with occasional deep fades
        mean = (low + high) / 2
        x = rng.uniform(low, high)
        cap = np.empty(n)
        for i in range(n):
            x += 0.05 * (mean - x) + rng.normal(0, 0.04 * (high - low))
            if rng.random() < 0.01:
                x = low + rng.uniform(0, 0.3) * (high - low)
            x = min(max(x, low), high)
            cap[i] = x

        # ---- held-out patterns (used only for generalisation tests) ----
    elif kind == "sawtooth":
        # slow linear ramp up, then an abrupt drop back to the minimum
        period = int(rng.integers(60, 120))
        phase = int(rng.integers(period))
        pos = ((np.arange(n) + phase) % period) / period
        cap = low + (high - low) * pos

    elif kind == "square":
        # fast periodic switching between high and low
        half = int(rng.integers(20, 60))
        phase = int(rng.integers(2 * half))
        cap = np.where(((np.arange(n) + phase) // half) % 2 == 0, high, low)

    elif kind == "outage":
        # stable high capacity with a few short deep outages
        cap = np.full(n, high, dtype=float)
        for _ in range(int(rng.integers(2, 5))):
            start = int(rng.integers(10, n - 40))
            cap[start:start + int(rng.integers(10, 25))] = low * 0.25

    elif kind == "ramp":
        # one long linear ramp, direction chosen at random
        a, b = (high, low) if rng.random() < 0.5 else (low, high)
        cap = np.linspace(a, b, n)
        
    else:
        raise ValueError(f"unknown scenario: {kind}")

    return np.clip(cap, 0.5, None)
