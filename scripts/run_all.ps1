# Reproduce everything end to end (PowerShell, from the repo root).
$ErrorActionPreference = "Stop"
foreach ($s in 0,1,2) {
  python -m src.agents.tabular_q --episodes 3000 --seed $s
  python -m src.agents.approx_q  --episodes 3000 --seed $s
  python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed $s
  python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed $s
}
python -m scripts.tune_aimd
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2
python -m src.evaluation.plots
foreach ($s in 0,1,2) {
  python -m src.agents.tabular_q --episodes 3000 --seed $s --robust
  python -m src.agents.approx_q  --episodes 3000 --seed $s --robust
  python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed $s --robust
  python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed $s --robust
}
python -m src.evaluation.generalization --episodes 10 --seeds 0 1 2