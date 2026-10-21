# RL-Based Network Congestion Control

A reinforcement-learning sender that adjusts its transmission rate over a bottleneck link with
fluctuating capacity, maximising throughput while limiting latency and packet loss.

## Results (5 capacity patterns x 20 episodes x 3 seeds)
| Method | Return | Utilisation | p95 latency (ms) | Loss |
|---|---|---|---|---|
| Random | 16.8 | 0.083 | 136 | 2.5% |
| Fixed rate | 166.4 | 0.937 | 369 | 5.9% |
| AIMD | 256.2 | 0.732 | 305 | 0.1% |
| Tabular Q | 363.1 | 0.976 | 130 | 0.0% |
| Approx. Q (linear) | 329.4 | 0.898 | 146 | 0.0% |
| DQN | 371.4 | 0.994 | 121 | 0.0% |
| PPO | 375.5 | 0.988 | 115 | 0.0% |

Generalisation (unseen patterns, buffers, RTT, noise, loss) is in `results/generalization_summary.csv`
and `results/figures/generalization.png`. Figures are in `results/figures/`.

## RL formulation
- **State:** last 8 monitoring intervals x 5 features (latency excess, latency gradient, sending ratio, loss rate, normalised throughput) = 40 dims
- **Actions:** 7 discrete rate multipliers {0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5}
- **Reward:** `utilisation - 0.3*queueing_delay/RTT - 3*loss - 0.1*|ln multiplier|`
- **Algorithms:** tabular Q-learning, linear approximate Q-learning, DQN, PPO (baselines: random, fixed rate, AIMD)

## Installation
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```
Dependencies: numpy, pandas, matplotlib, gymnasium, stable-baselines3, torch, pytest.

## Run (from the repo root)
```bash
python -m src.agents.tabular_q --episodes 3000 --seed 0          # add --robust for noise-randomised training
python -m src.agents.approx_q  --episodes 3000 --seed 0
python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed 0
python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed 0
python -m scripts.tune_aimd                                       # AIMD tuning on held-out seeds
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2     # main comparison (+ robust agents if trained)
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2 --rand-loss 0.01
python -m src.evaluation.generalization --episodes 10 --seeds 0 1 2
python -m src.evaluation.plots                                    # figures -> results/figures/
```

## Reproduce everything
`bash scripts/run_all.sh` (Linux/macOS/Git Bash) or `.\scripts\run_all.ps1` (PowerShell).
Training is the slow part; the committed `models/` and `results/` let you skip it and only run the
evaluation and plotting commands above.

## Layout
```
src/config.py                 actions, scenarios, environment and reward parameters
src/envs/                     capacity generators and the Gymnasium environment
src/baselines/                random, fixed-rate and AIMD policies
src/agents/                   tabular Q, linear approximate Q, DQN/PPO training
src/evaluation/               evaluate.py, generalization.py, plots.py
scripts/                      tune_aimd.py, run_all scripts
tests/                        simulator tests
models/                       trained agents (3 seeds; *_robust_* = noise-randomised training)
results/                      metrics, traces, training logs, figures
report/                       2-page report (LaTeX source and PDF)
```
