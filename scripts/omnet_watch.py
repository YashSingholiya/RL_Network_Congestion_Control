"""Run ONE episode in OMNeT++ with a chosen controller, and either watch it in the OMNeT++ window or record it.

  # watch in the OMNeT++ graphical window (Qtenv); click "Run" or "Fast" in the window to start the simulation
  python scripts/omnet_watch.py --controller aimd --pattern markov --gui

  # record result vectors (.vec) and the eventlog (.elog) for the OMNeT++ IDE (Analysis tool / Sequence chart)
  python scripts/omnet_watch.py --controller tabular_q --pattern markov --record --steps 80

  # plain run, prints the episode summary
  python scripts/omnet_watch.py --controller fixed --pattern markov

Run from the repository root, in the same Linux environment where OMNeT++ was built.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
from src.config import EnvConfig, SCENARIOS, HELDOUT_SCENARIOS
from src.envs.omnet_env import OmnetCongestionEnv, omnet_available, default_exe
from src.evaluation.evaluate import load_policies, run_episode


class Paced:
    """Wraps a controller and waits `delay` seconds per decision, so the OMNeT++ window runs in (almost) real time."""

    def __init__(self, inner, delay):
        self.inner, self.delay = inner, delay
        self.uses_env = getattr(inner, "uses_env", False)
        self.reset_options = getattr(inner, "reset_options", {})

    def act(self, obs, env=None):
        time.sleep(self.delay)
        return self.inner.act(obs, env) if self.uses_env else self.inner.act(obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="aimd",
                   help="fixed | aimd | random | tabular_q | approx_q | dqn | ppo (+ _robust variants)")
    p.add_argument("--agent-seed", type=int, default=0, help="which trained copy of a learned agent to load")
    p.add_argument("--pattern", default="markov", choices=SCENARIOS + HELDOUT_SCENARIOS)
    p.add_argument("--seed", type=int, default=10_000, help="episode seed (10000 = the episode used in the report)")
    p.add_argument("--steps", type=int, default=400, help="number of 50 ms steps (400 = full 20 s episode)")
    p.add_argument("--gui", action="store_true", help="open the OMNeT++ window (Qtenv)")
    p.add_argument("--record", action="store_true", help="record .vec/.sca/.elog files into omnetpp/results/")
    p.add_argument("--delay", type=float, default=None,
                   help="seconds of wall-clock time per 50 ms step (default: 0.05 with --gui = real time, 0 otherwise)")
    p.add_argument("--latency", choices=["last", "mean", "queue"], default="last")
    a = p.parse_args()

    if not omnet_available():
        raise SystemExit(f"OMNeT++ model not found at {default_exe()} (see README.md)")
    policies = load_policies(a.controller, [a.agent_seed])
    if not policies:
        raise SystemExit(f"could not load controller '{a.controller}' (is the model file in models/?)")
    policy = policies[0][1]
    delay = a.delay if a.delay is not None else (0.05 if a.gui else 0.0)
    if delay > 0:
        policy = Paced(policy, delay)

    cfg = EnvConfig(lat_noise=0.0, episode_len=a.steps)
    env = OmnetCongestionEnv(cfg, scenario=a.pattern, ui="Qtenv" if a.gui else "Cmdenv",
                             config_name="Record" if a.record else "Bridge", latency=a.latency,
                             timeout=3600 if a.gui else 120)         # in the GUI you may pause for a long time
    if a.gui:
        print("The OMNeT++ window opens now. Click 'Run' (or 'Fast') in its toolbar to start the simulation; "
              "the episode then plays in about real time (20 s). The window stays open at the end: close it to finish.", flush=True)
    m, trace = run_episode(env, policy, seed=a.seed)
    print(f"\nfinished: {a.controller} on '{a.pattern}' (seed {a.seed}, {a.steps} steps)")
    print({k: round(float(v), 3) for k, v in m.items()})
    if a.record:
        print("\nRecorded in omnetpp/results/ (open them with the OMNeT++ IDE: File > Open, Analysis Tool / Sequence Chart).")


if __name__ == "__main__":
    main()