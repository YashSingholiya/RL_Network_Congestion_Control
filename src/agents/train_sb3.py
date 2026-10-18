"""Train DQN or PPO (Stable-Baselines3) on the congestion-control environment.

DQN: experience replay + fixed target network (Lec 24).
PPO: actor-critic, GAE, clipped surrogate objective (Lec 24).
Run:  python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed 0
"""
import argparse
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from src.config import robust_train_cfg

from src.envs.congestion_env import CongestionEnv

GAMMA = 0.95
NET = dict(net_arch=[64, 64])


def build(algo, env, seed):
    if algo == "dqn":
        return DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=100_000,
                   learning_starts=2_000, batch_size=64, gamma=GAMMA, train_freq=4,
                   target_update_interval=1_000, exploration_fraction=0.3,
                   exploration_final_eps=0.05, policy_kwargs=NET, seed=seed, verbose=0)
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=128,
               n_epochs=10, gamma=GAMMA, gae_lambda=0.95, clip_range=0.2,
               ent_coef=0.01, policy_kwargs=NET, seed=seed, verbose=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robust", action="store_true", help="randomise latency noise during training")
    a = p.parse_args()

    tag = "_robust" if a.robust else ""
    logdir = f"results/logs/{a.algo}{tag}_s{a.seed}"
    os.makedirs(logdir, exist_ok=True)
    n_envs = 4 if a.algo == "ppo" else 1
    env_kwargs = dict(cfg=robust_train_cfg()) if a.robust else {}
    env = make_vec_env(CongestionEnv, n_envs=n_envs, seed=a.seed, monitor_dir=logdir, env_kwargs=env_kwargs)
    model = build(a.algo, env, a.seed)
    model.learn(total_timesteps=a.timesteps, progress_bar=False)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{a.algo}{tag}_s{a.seed}")
    print(f"saved models/{a.algo}{tag}_s{a.seed}.zip")


if __name__ == "__main__":
    main()
