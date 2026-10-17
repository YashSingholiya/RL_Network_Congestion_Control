"""Central configuration: action set, scenarios, and environment parameters."""
from dataclasses import dataclass
from typing import Optional, Tuple

# Discrete action set: multiplicative change applied to the sending rate each step.
ACTIONS = [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]
HOLD_IDX = ACTIONS.index(1.0)

# Capacity patterns used for the "fluctuating network capacity".
SCENARIOS = ["constant", "step", "sine", "markov", "trace"]

# Held-out capacity patterns: NEVER sampled during training, used only in generalisation tests.
HELDOUT_SCENARIOS = ["sawtooth", "square", "outage", "ramp"]

N_FEATURES = 5  # features per monitoring interval (see congestion_env.py)


@dataclass
class EnvConfig:
    # --- simulation ---
    dt: float = 0.05                 # monitoring interval (MI) in seconds
    episode_len: int = 400           # steps per episode (= 20 s simulated)
    history: int = 8                 # number of past MIs stacked in the observation

    # --- domain randomisation (sampled at every reset) ---
    rtt_range: Tuple[float, float] = (0.04, 0.15)        # base RTT, seconds
    cap_high_range: Tuple[float, float] = (8.0, 20.0)    # peak capacity, Mbps
    cap_low_frac_range: Tuple[float, float] = (0.2, 0.6) # trough = frac * peak
    buffer_bdp_range: Tuple[float, float] = (1.0, 4.0)   # router buffer in BDPs

    init_rate_range: Tuple[float, float] = (0.1, 1.5)    # initial rate = U(range) x capacity

    # --- imperfections ---
    rand_loss: float = 0.0           # non-congestive random loss probability
    lat_noise: float = 0.02          # latency noise std, as a fraction of base RTT
    lat_noise_range: Optional[Tuple[float, float]] = None   # if set: noise std ~ U(range) per episode (robust training)

    # --- reward weights: r = w_thr*util - w_delay*qdelay/rtt - w_loss*loss - w_smooth*|ln m| ---
    w_thr: float = 1.0
    w_delay: float = 0.3
    w_loss: float = 3.0
    w_smooth: float = 0.1

    # --- rate limits ---
    min_rate: float = 0.1            # Mbps
    max_rate_mult: float = 1.5       # max rate = mult * peak capacity

    # --- fixed scenario (None = random scenario every episode) ---
    scenario: Optional[str] = None
    
# Noise-randomised training: latency noise std sampled per episode from this range
# (fraction of base RTT). Evaluation suites use 0.10 and 0.20, i.e. at / beyond its edge.
ROBUST_LAT_NOISE_RANGE = (0.01, 0.08)


def robust_train_cfg():
    return EnvConfig(lat_noise_range=ROBUST_LAT_NOISE_RANGE)