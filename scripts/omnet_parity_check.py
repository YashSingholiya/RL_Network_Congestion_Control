"""Parity check between the Python fluid simulator and the OMNeT++ packet-level simulator.

Runs the non-learning controllers (Fixed rate, AIMD) on the SAME links in both simulators and compares the
episode metrics and the second-by-second traces. Needs the compiled model (README.md).

    python scripts/omnet_parity_check.py [--episodes 3] [--patterns constant markov step]

Exit code 0 when every comparison is within the tolerances below (suggested values; adjust with the options).
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.config import EnvConfig
from src.envs.congestion_env import CongestionEnv
from src.envs.omnet_env import OmnetCongestionEnv, omnet_available, default_exe
from src.baselines.simple import FixedRatePolicy
from src.baselines.aimd import AIMDPolicy
from src.evaluation.evaluate import run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--patterns", nargs="+", default=["constant", "markov", "step"])
    p.add_argument("--tol-util", type=float, default=0.05, help="max |util_omnet - util_python|")
    p.add_argument("--tol-lat", type=float, default=0.25, help="max relative difference of the mean latency")
    p.add_argument("--min-corr", type=float, default=0.85, help="min correlation of the per-step queue traces")
    a = p.parse_args()
    if not omnet_available():
        raise SystemExit(f"OMNeT++ model not found at {default_exe()} (see README.md)")

    cfg = EnvConfig(lat_noise=0.0)                  # no artificial noise on either side
    rows = []
    for sc in a.patterns:
        py, om = CongestionEnv(cfg, scenario=sc), OmnetCongestionEnv(cfg, scenario=sc)
        for name, make in (("fixed", lambda: FixedRatePolicy(1.0)), ("aimd", AIMDPolicy)):
            for ep in range(a.episodes):
                (m1, t1), (m2, t2) = run_episode(py, make(), 10_000 + ep), run_episode(om, make(), 10_000 + ep)
                qc = float(np.corrcoef(t1["queue"], t2["queue"])[0, 1]) if t1["queue"].std() > 0 and t2["queue"].std() > 0 else 1.0
                rows.append(dict(pattern=sc, method=name, ep=ep,
                                 util_py=m1["util"], util_om=m2["util"], lat_py=m1["avg_lat_ms"], lat_om=m2["avg_lat_ms"],
                                 loss_py=m1["loss"], loss_om=m2["loss"], ret_py=m1["reward"], ret_om=m2["reward"], queue_corr=qc))
    df = pd.DataFrame(rows)
    g = df.groupby(["pattern", "method"]).mean(numeric_only=True).drop(columns="ep")
    g["d_util"] = g.util_om - g.util_py
    g["rel_d_lat"] = (g.lat_om - g.lat_py) / g.lat_py
    pd.set_option("display.width", 220)
    print(g.round(3).to_string())
    bad = g[(g.d_util.abs() > a.tol_util) | (g.rel_d_lat.abs() > a.tol_lat) | (g.queue_corr < a.min_corr)]
    print("\nPARITY OK" if bad.empty else f"\nOUTSIDE TOLERANCE:\n{bad[['d_util', 'rel_d_lat', 'queue_corr']].round(3).to_string()}")
    sys.exit(0 if bad.empty else 1)


if __name__ == "__main__":
    main()
