"""Gymnasium environment backed by the OMNeT++ packet-level model in ``omnetpp/``.

It has the SAME interface as ``CongestionEnv`` (same observation, actions, reward, ``info`` keys and the
attributes the evaluation code and dashboard read), so every controller can run unchanged in either simulator.

How it works
------------
* ``reset(seed)`` first lets the Python fluid environment draw the episode (pattern, capacity array, base RTT,
  buffer size, initial rate), so both simulators face exactly the same link ("config parity").
* It then starts the OMNeT++ executable, which connects back over TCP, receives the episode as one text line
  (CONFIG ...) and waits.
* ``step(action)`` turns the action into a new sending rate (same clipping as the fluid model), sends it to OMNeT++,
  which simulates 50 ms of packets and returns raw statistics (OBS ...). The five features and the reward are
  computed here with the same formulas as in ``congestion_env.py``.

``latency`` selects what the agent sees as latency: "last" (default: RTT of the newest ACK), "mean" (mean RTT of the
ACKs of the interval) or "queue" (ablation: base RTT + queue/capacity at the end of the interval, as in the fluid model).

Differences from the fluid model (intentional, they are the point of the packet-level check): data moves in
1500-byte packets, latency is measured from ACKs (so it lags the queue by about one RTT) and is quantised by the
number of packets per interval; the buffer is at least 2 packets.
"""
import dataclasses
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import ACTIONS, EnvConfig
from src.envs.congestion_env import CongestionEnv, FEAT_LOW, FEAT_HIGH

REPO_ROOT = Path(__file__).resolve().parents[2]
OMNET_DIR = REPO_ROOT / "omnetpp"
PACKET_BITS = 12_000                      # 1500-byte data packets


def default_exe():
    """Path of the compiled OMNeT++ model (override with the OMNETPP_CONGESTION_EXE environment variable)."""
    env = os.environ.get("OMNETPP_CONGESTION_EXE")
    if env:
        return Path(env)
    for name in ("congestion", "congestion_dbg", "congestion.exe"):
        if (OMNET_DIR / name).exists():
            return OMNET_DIR / name
    return OMNET_DIR / "congestion"


def omnet_available():
    return default_exe().exists()


class FeatureTracker:
    """The five observation features, computed exactly as in CongestionEnv.step()."""

    def __init__(self, base_rtt):
        self.reset(base_rtt)

    def reset(self, base_rtt):
        self.base_rtt = base_rtt
        self.min_lat = np.inf
        self.prev_lat = base_rtt
        self.max_thr = 1e-6

    def update(self, rate, thr, latency, loss):
        self.min_lat = min(self.min_lat, latency)
        self.max_thr = max(self.max_thr, thr)
        feat = np.array([
            np.clip(latency / self.min_lat - 1.0, 0.0, 5.0),
            np.clip((latency - self.prev_lat) / self.base_rtt, -1.0, 1.0),
            np.clip(rate / max(thr, 1e-3), 0.0, 3.0),
            np.clip(loss, 0.0, 1.0),
            np.clip(thr / self.max_thr, 0.0, 1.0),
        ], dtype=np.float32)
        self.prev_lat = latency
        return feat


def step_reward(cfg, thr, cap, qdelay, base_rtt, loss, mult):
    """Reward exactly as in CongestionEnv.step()."""
    return (cfg.w_thr * (thr / cap)
            - cfg.w_delay * min(qdelay / base_rtt, 5.0)
            - cfg.w_loss * loss
            - cfg.w_smooth * abs(np.log(mult)))


class OmnetCongestionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, scenario=None, exe=None, ini=None, config_name="Bridge",
                 packet_bits=PACKET_BITS, timeout=120.0, launcher=None, latency="last", log_dir=None,
                 ui=None, extra_args=(), gui_linger=3600.0):
        super().__init__()
        cfg = cfg or EnvConfig()
        if scenario is not None:
            cfg = dataclasses.replace(cfg, scenario=scenario)
        self.cfg = cfg
        self.ref = CongestionEnv(cfg)               # draws the episode: identical link in both simulators
        k = cfg.history
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=np.tile(FEAT_LOW, k), high=np.tile(FEAT_HIGH, k), dtype=np.float32)
        self.exe = Path(exe) if exe else default_exe()
        self.ini = Path(ini) if ini else OMNET_DIR / "omnetpp.ini"
        self.config_name, self.packet_bits, self.timeout = config_name, packet_bits, timeout
        self.launcher, self.latency_mode, self.log_dir = launcher, latency, log_dir
        self.ui = ui or os.environ.get("OMNETPP_UI", "Cmdenv")   # "Qtenv" opens the OMNeT++ graphical window
        self.extra_args = tuple(extra_args)
        self.gui_linger = gui_linger                    # seconds the OMNeT++ window may stay open after the episode
        self._proc = self._conn = self._srv = self._log = None
        self._rbuf = b""
        self._log_path = None
        self.t = 0

    # ----------------------------------------------------------------- process / socket handling
    def _log_tail(self, n=15):
        try:
            return "\n".join(Path(self._log_path).read_text(errors="replace").splitlines()[-n:])
        except Exception:
            return "(no log)"

    def _send(self, line):
        self._conn.sendall((line + "\n").encode())

    def _readline(self):
        while b"\n" not in self._rbuf:
            try:
                chunk = self._conn.recv(65536)
            except socket.timeout:
                raise RuntimeError("Timed out waiting for OMNeT++.\n" + self._log_tail())
            if not chunk:
                raise RuntimeError("OMNeT++ closed the connection unexpectedly.\n" + self._log_tail())
            self._rbuf += chunk
        line, self._rbuf = self._rbuf.split(b"\n", 1)
        return line.decode().strip()

    def _start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self._srv.settimeout(0.5)
        port = self._srv.getsockname()[1]
        fd, self._log_path = tempfile.mkstemp(prefix="omnet_", suffix=".log", dir=self.log_dir)
        self._log = os.fdopen(fd, "w")
        if self.launcher is not None:
            self._proc = self.launcher(port)
        else:
            if not self.exe.exists():
                raise FileNotFoundError(
                    f"OMNeT++ model not found at {self.exe}. Build it first (see README.md) "
                    "or set OMNETPP_CONGESTION_EXE.")
            cmd = [str(self.exe), "-u", self.ui, "-f", str(self.ini), "-c", self.config_name, "-r", "0",
                   "-n", str(self.ini.parent), f"--**.bridge.port={port}", *self.extra_args]
            self._proc = subprocess.Popen(cmd, cwd=str(self.ini.parent), stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.timeout
        while self._conn is None:
            try:
                self._conn, _ = self._srv.accept()
            except socket.timeout:
                if self._proc is not None and hasattr(self._proc, "poll") and self._proc.poll() is not None:
                    raise RuntimeError("OMNeT++ exited before connecting.\n" + self._log_tail())
                if time.time() > deadline:
                    raise RuntimeError("OMNeT++ did not connect in time.\n" + self._log_tail())
        self._conn.settimeout(self.timeout)
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rbuf = b""
        if self._readline() != "HELLO":
            raise RuntimeError("Unexpected first message from OMNeT++.\n" + self._log_tail())
        n = self.cfg.episode_len
        caps = " ".join(f"{c:.9g}" for c in self.cap[:n])
        self._send(f"CONFIG {self.base_rtt:.9g} {self.buffer_bits:.9g} {self.packet_bits:.9g} "
                   f"{self.cfg.dt:.9g} {n} {n} {caps}")
        if self._readline() != "READY":
            raise RuntimeError("OMNeT++ did not report READY.\n" + self._log_tail())

    def close(self):
        for s in (self._conn, self._srv):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self._conn = self._srv = None
        p = self._proc
        if p is not None and hasattr(self, "_proc"):
            try:
                if p.poll() is None:
                    linger = 5 if self.ui == "Cmdenv" else self.gui_linger   # GUI: wait until the window is closed
                    try:
                        p.wait(timeout=linger)
                    except KeyboardInterrupt:
                        p.terminate()
                    except Exception:
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except Exception:
                            p.kill()
            except Exception:
                pass
        self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._log_path and os.path.exists(self._log_path):
            try:
                os.remove(self._log_path)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.ref.reset(seed=seed, options=options)        # same random link as the fluid simulator
        r = self.ref
        self.scenario, self.base_rtt, self.cap = r.scenario, r.base_rtt, r.cap
        self.mean_cap, self.max_rate = r.mean_cap, r.max_rate
        self.buffer_bits = max(r.qmax * 1e6, 2 * self.packet_bits)     # at least 2 packets
        self.qmax = self.buffer_bits / 1e6                              # Mbit (effective buffer)
        self.rate = r.rate
        self.t = 0
        self.tracker = FeatureTracker(self.base_rtt)
        self.hist = np.tile(np.array([0, 0, 1, 0, 0], dtype=np.float32), (self.cfg.history, 1))
        self._start()
        return self.hist.flatten().astype(np.float32), {"scenario": self.scenario}

    def step(self, action):
        c = self.cfg
        mult = ACTIONS[int(action)]
        self.rate = float(np.clip(self.rate * mult, c.min_rate, self.max_rate))
        self._send(f"RATE {self.rate:.9g}")
        tok = self._readline().split()
        if len(tok) != 11 or tok[0] != "OBS":
            raise RuntimeError(f"Unexpected message from OMNeT++: {' '.join(tok)}\n" + self._log_tail())
        (sent, served, dropped, queue_bits, rtt_last, rtt_mean, acks, cap) = map(float, tok[2:10])
        done = tok[10] == "1"
        thr = served / 1e6 / c.dt                                          # Mbps
        queue = queue_bits / 1e6                                           # Mbit
        qdelay = queue / cap
        if self.latency_mode == "queue":                                   # ablation: instantaneous, like the fluid model
            lat = self.base_rtt + qdelay
        else:
            rtt = rtt_last if self.latency_mode == "last" else rtt_mean
            lat = rtt if rtt >= 0 else self.tracker.prev_lat               # no ACK seen yet: keep the last value
        loss = dropped / sent if sent > 0 else 0.0
        feat = self.tracker.update(self.rate, thr, lat, loss)
        reward = step_reward(c, thr, cap, qdelay, self.base_rtt, loss, mult)
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = feat
        self.t += 1
        info = dict(throughput=thr, latency=lat, loss=loss, queue=queue, capacity=cap, rate=self.rate,
                    util=thr / cap, qdelay=qdelay, arr_eff=sent / 1e6, arrivals=sent / 1e6, service=served / 1e6,
                    overflow=dropped / 1e6, rand_lost=0.0, acks=acks, rtt_mean=rtt_mean)
        if done:
            if self.ui != "Cmdenv":
                print("Episode finished. Close the OMNeT++ window to continue.", flush=True)
            self.close()                                                    # reaps the finished OMNeT++ process
        return self.hist.flatten().astype(np.float32), float(reward), False, done, info