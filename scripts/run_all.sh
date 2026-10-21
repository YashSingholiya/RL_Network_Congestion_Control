#!/usr/bin/env bash
# Reproduce everything end to end (from the repo root).
set -e
for SEED in 0 1 2; do
  python -m src.agents.tabular_q --episodes 3000 --seed $SEED
  python -m src.agents.approx_q  --episodes 3000 --seed $SEED
  python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed $SEED
  python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed $SEED
done
python -m scripts.tune_aimd
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2
python -m src.evaluation.plots
for SEED in 0 1 2; do
  python -m src.agents.tabular_q --episodes 3000 --seed $SEED --robust
  python -m src.agents.approx_q  --episodes 3000 --seed $SEED --robust
  python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed $SEED --robust
  python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed $SEED --robust
done
python -m src.evaluation.generalization --episodes 10 --seeds 0 1 2