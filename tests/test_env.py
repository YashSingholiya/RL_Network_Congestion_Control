"""Sanity tests for the simulator. Run: pytest -q"""
import numpy as np
from src.config import SCENARIOS, HELDOUT_SCENARIOS, ACTIONS, HOLD_IDX, EnvConfig
from src.envs.congestion_env import CongestionEnv


def test_spaces_and_length():
    env = CongestionEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    n, done = 0, False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
        n, done = n + 1, term or trunc
    assert n == EnvConfig().episode_len


def test_flow_conservation():
    """arrivals = delivered + overflow drops + final queue."""
    for sc in SCENARIOS + HELDOUT_SCENARIOS:
        env = CongestionEnv(scenario=sc)
        env.reset(seed=1)
        arr = srv = ovf = 0.0
        done = False
        while not done:
            _, _, _, trunc, info = env.step(env.action_space.sample())
            arr += info["arr_eff"]; srv += info["service"]; ovf += info["overflow"]
            done = trunc
        assert abs(arr - (srv + ovf + env.queue)) < 1e-6


def test_matched_rate_is_good():
    env = CongestionEnv(scenario="constant")
    env.reset(seed=2, options={"init_rate_frac": 1.0})
    utils, losses = [], []
    for _ in range(100):
        _, _, _, _, info = env.step(HOLD_IDX)
        utils.append(info["util"]); losses.append(info["loss"])
    assert np.mean(utils) > 0.99 and max(losses) == 0.0


def test_overload_causes_delay_and_loss():
    env = CongestionEnv(scenario="constant")
    env.reset(seed=3, options={"init_rate_frac": 2.0})
    last = None
    for _ in range(200):
        _, _, _, _, last = env.step(HOLD_IDX)
    assert last["loss"] > 0.2 and last["qdelay"] > 0


def test_determinism():
    runs = []
    for _ in range(2):
        env = CongestionEnv()
        env.reset(seed=5)
        runs.append([env.step(3)[1] for _ in range(20)])
    assert runs[0] == runs[1]
