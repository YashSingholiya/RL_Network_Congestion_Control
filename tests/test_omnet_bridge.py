"""Tests for the OMNeT++ integration.

1-3 need no OMNeT++: a small Python thread plays the role of the OMNeT++ process (same text protocol and the same
fluid equations as the Python simulator), so they check the bridge, the feature/reward code and the episode setup.
4   runs the real compiled OMNeT++ model and is skipped when it has not been built.
Run: python -m pytest -q tests/test_omnet_bridge.py
"""
import socket
import threading

import numpy as np
import pytest

from src.config import ACTIONS, HOLD_IDX, EnvConfig
from src.envs.congestion_env import CongestionEnv
from src.envs.omnet_env import OmnetCongestionEnv, FeatureTracker, step_reward, omnet_available


class FakeProc:
    def __init__(self, thread):
        self.thread = thread
        self.wait_timeouts = []

    def poll(self):
        return None if self.thread.is_alive() else 0

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.thread.join(timeout)

    def terminate(self):
        pass

    kill = terminate


def mock_launcher(port):
    """Plays the OMNeT++ side of the protocol using the fluid equations."""
    def run():
        s = socket.create_connection(("127.0.0.1", port))
        f = s.makefile("rwb", buffering=0)
        f.write(b"HELLO\n")
        tok = f.readline().decode().split()
        base_rtt, buf_bits, pkt_bits, dt = map(float, tok[1:5])
        steps, ncap = int(tok[5]), int(tok[6])
        cap = list(map(float, tok[7:7 + ncap]))
        f.write(b"READY\n")
        q = 0.0
        for k in range(steps):
            rate = float(f.readline().decode().split()[1])
            sent = rate * 1e6 * dt
            avail = q + sent
            served = min(avail, cap[k] * 1e6 * dt)
            q = avail - served
            drop = max(0.0, q - buf_bits)
            q = min(q, buf_bits)
            rtt = base_rtt + q / (cap[k] * 1e6)
            done = 1 if k == steps - 1 else 0
            f.write(f"OBS {k + 1} {sent:.9g} {served:.9g} {drop:.9g} {q:.9g} {rtt:.9g} {rtt:.9g} 10 {cap[k]:.9g} {done}\n".encode())
        s.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return FakeProc(t)


def test_protocol_and_episode_shape():
    env = OmnetCongestionEnv(scenario="markov", launcher=mock_launcher, timeout=20)
    obs, info = env.reset(seed=3)
    assert obs.shape == (40,) and env.observation_space.contains(obs)
    n, done, last = 0, False, None
    while not done:
        obs, r, term, trunc, last = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r) and not term
        assert last["util"] <= 1 + 1e-9
        n, done = n + 1, trunc
    assert n == EnvConfig().episode_len
    for key in ("throughput", "latency", "loss", "queue", "capacity", "rate", "util", "qdelay", "arrivals", "overflow"):
        assert key in last
    env.close()


def test_features_and_reward_match_the_fluid_simulator():
    """Feeding the fluid simulator's raw values to the bridge-side code must reproduce its features and reward."""
    cfg = EnvConfig(lat_noise=0.0)
    ref = CongestionEnv(cfg, scenario="step")
    ref.reset(seed=11)
    trk = FeatureTracker(ref.base_rtt)
    rng = np.random.default_rng(0)
    for _ in range(cfg.episode_len):
        a = int(rng.integers(len(ACTIONS)))
        obs, r, _, _, info = ref.step(a)
        feat = trk.update(info["rate"], info["throughput"], info["latency"], info["loss"])
        r2 = step_reward(cfg, info["throughput"], info["capacity"], info["qdelay"], ref.base_rtt, info["loss"], ACTIONS[a])
        assert np.allclose(feat, obs[-5:], atol=1e-5)
        assert abs(r - r2) < 1e-9


@pytest.mark.parametrize("scenario", ["constant", "markov", "trace"])
def test_bridge_reproduces_fluid_results_end_to_end(scenario):
    """With a mock that implements the fluid equations, the OMNeT++ environment must equal CongestionEnv."""
    cfg = EnvConfig(lat_noise=0.0)
    ref = CongestionEnv(cfg, scenario=scenario)
    env = OmnetCongestionEnv(cfg, scenario=scenario, launcher=mock_launcher, timeout=20)
    obs_ref, _ = ref.reset(seed=5)
    obs_om, _ = env.reset(seed=5)
    assert np.allclose(obs_ref, obs_om)
    rng = np.random.default_rng(1)
    for _ in range(cfg.episode_len):
        a = int(rng.integers(len(ACTIONS)))
        o1, r1, _, t1, i1 = ref.step(a)
        o2, r2, _, t2, i2 = env.step(a)
        assert t1 == t2
        assert np.allclose(o1, o2, atol=1e-4), (o1[-5:], o2[-5:])
        assert abs(r1 - r2) < 1e-6
    env.close()


@pytest.mark.skipif(not omnet_available(), reason="OMNeT++ model not built (see README.md)")
def test_real_omnet_model_basic_behaviour():
    cfg = EnvConfig(lat_noise=0.0)

    def run(frac, seed):
        env = OmnetCongestionEnv(cfg, scenario="constant")
        env.reset(seed=seed, options={"init_rate_frac": frac})
        rows, done = [], False
        while not done:
            _, r, _, done, info = env.step(HOLD_IDX)
            rows.append((info, r))
        return rows, env

    rows, env = run(1.0, 2)                                     # matched rate: full link, no loss
    util = np.mean([i["util"] for i, _ in rows])
    assert util > 0.95 and sum(i["overflow"] for i, _ in rows) == 0.0
    sent = sum(i["arrivals"] for i, _ in rows)                  # conservation: sent = served + dropped + queued
    served = sum(i["service"] for i, _ in rows)
    queued = rows[-1][0]["queue"]
    assert abs(sent - served - queued) < 0.05 * max(sent, 1e-9)
    rows2, _ = run(1.0, 2)                                      # determinism
    assert [round(r, 9) for _, r in rows] == [round(r, 9) for _, r in rows2]
    over, _ = run(1.6, 2)                                       # overload: queue fills, packets are dropped
    assert np.mean([i["loss"] for i, _ in over]) > 0.1


class LingeringProc(FakeProc):
    """Like a GUI window: stays alive until wait() returns (the user closes the window)."""

    def __init__(self, thread):
        super().__init__(thread)
        self.released = False

    def poll(self):
        return 0 if self.released else None

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.released = True


def test_gui_window_is_not_killed_after_the_episode():
    """With a GUI the environment must wait for the user to close the window, not terminate it after 5 seconds."""
    procs = []

    def launcher(port):
        procs.append(LingeringProc(mock_launcher(port).thread))
        return procs[-1]

    env = OmnetCongestionEnv(EnvConfig(episode_len=5), scenario="constant", launcher=launcher, ui="Qtenv", gui_linger=1234, timeout=20)
    env.reset(seed=1)
    done = False
    while not done:
        _, _, _, done, _ = env.step(HOLD_IDX)
    assert 1234 in procs[0].wait_timeouts
    assert 5 not in procs[0].wait_timeouts