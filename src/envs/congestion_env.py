"""Gymnasium environment: one sender, one bottleneck link with a drop-tail queue.

Fluid model, one step = one monitoring interval (MI) of `dt` seconds.

State  (history * 5 dims, newest MI last). Per-MI features:
    f0 latency excess      = latency / min_latency_seen - 1          in [0, 5]
    f1 latency gradient    = (latency - prev_latency) / base_rtt     in [-1, 1]
    f2 sending ratio       = send_rate / throughput                  in [0, 3]
    f3 loss rate           = lost / sent                             in [0, 1]
    f4 normalised thr.     = throughput / max_throughput_seen        in [0, 1]
Action (Discrete): index into ACTIONS = multiplicative change of the sending rate.
Reward: w_thr*utilisation - w_delay*queueing_delay/base_rtt - w_loss*loss - w_smooth*|ln m|
"""
import dataclasses
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import ACTIONS, SCENARIOS, N_FEATURES, EnvConfig
from src.envs.capacity import make_capacity

FEAT_LOW = np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
FEAT_HIGH = np.array([5.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float32)


class CongestionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, scenario=None):
        super().__init__()
        cfg = cfg or EnvConfig()
        if scenario is not None:
            cfg = dataclasses.replace(cfg, scenario=scenario)
        self.cfg = cfg
        self.action_space = spaces.Discrete(len(ACTIONS))
        k = cfg.history
        self.observation_space = spaces.Box(
            low=np.tile(FEAT_LOW, k), high=np.tile(FEAT_HIGH, k), dtype=np.float32
        )

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c, rng = self.cfg, self.np_random

        self.scenario = c.scenario or str(rng.choice(SCENARIOS))
        self.base_rtt = float(rng.uniform(*c.rtt_range))
        high = float(rng.uniform(*c.cap_high_range))
        low = high * float(rng.uniform(*c.cap_low_frac_range))
        self.cap = make_capacity(self.scenario, c.episode_len + 1, c.dt, low, high, rng)
        self.mean_cap = float(self.cap[: c.episode_len].mean())
        bdp = self.mean_cap * self.base_rtt                       # Mbit
        self.qmax = float(rng.uniform(*c.buffer_bdp_range)) * bdp  # Mbit
        self.max_rate = c.max_rate_mult * high

        u = float(rng.uniform(*c.init_rate_range))  # always drawn -> identical RNG stream across policies
        frac = (options or {}).get("init_rate_frac")
        init = frac * self.mean_cap if frac is not None else u * self.cap[0]
        self.rate = float(np.clip(init, c.min_rate, self.max_rate))

        # drawn last, and only when randomisation is on -> default RNG stream is unchanged
        self.lat_noise = float(rng.uniform(*c.lat_noise_range)) if c.lat_noise_range else c.lat_noise

        self.queue = 0.0
        self.t = 0
        self.min_lat = np.inf
        self.prev_lat = self.base_rtt
        self.max_thr = 1e-6
        init_feat = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.hist = np.tile(init_feat, (c.history, 1))
        return self.hist.flatten().astype(np.float32), {"scenario": self.scenario}

    # ------------------------------------------------------------------- step
    def step(self, action):
        c, rng = self.cfg, self.np_random
        mult = ACTIONS[int(action)]
        self.rate = float(np.clip(self.rate * mult, c.min_rate, self.max_rate))
        cap = float(self.cap[self.t])

        arrivals = self.rate * c.dt                    # Mbit offered this MI
        rand_lost = arrivals * c.rand_loss             # non-congestive loss
        arr_eff = arrivals - rand_lost
        avail = self.queue + arr_eff
        service = min(avail, cap * c.dt)               # Mbit delivered this MI
        q = avail - service
        overflow = max(0.0, q - self.qmax)             # drop-tail losses
        self.queue = min(q, self.qmax)

        thr = service / c.dt                           # Mbps
        qdelay = self.queue / cap                      # seconds
        lat = self.base_rtt + qdelay + float(rng.normal(0, self.lat_noise * self.base_rtt))
        lat = max(lat, 0.5 * self.base_rtt)
        loss = (overflow + rand_lost) / arrivals if arrivals > 1e-12 else 0.0

        # ---- observation features
        self.min_lat = min(self.min_lat, lat)
        self.max_thr = max(self.max_thr, thr)
        feat = np.array([
            np.clip(lat / self.min_lat - 1.0, 0.0, 5.0),
            np.clip((lat - self.prev_lat) / self.base_rtt, -1.0, 1.0),
            np.clip(self.rate / max(thr, 1e-3), 0.0, 3.0),
            np.clip(loss, 0.0, 1.0),
            np.clip(thr / self.max_thr, 0.0, 1.0),
        ], dtype=np.float32)
        self.prev_lat = lat
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = feat

        # ---- reward
        util = thr / cap
        reward = (c.w_thr * util
                  - c.w_delay * min(qdelay / self.base_rtt, 5.0)
                  - c.w_loss * loss
                  - c.w_smooth * abs(np.log(mult)))

        self.t += 1
        truncated = self.t >= c.episode_len
        info = dict(throughput=thr, latency=lat, loss=loss, queue=self.queue,
                    capacity=cap, rate=self.rate, util=util, qdelay=qdelay,
                    arr_eff=arr_eff, arrivals=arrivals, service=service,
                    overflow=overflow, rand_lost=rand_lost)
        return self.hist.flatten().astype(np.float32), float(reward), False, truncated, info
