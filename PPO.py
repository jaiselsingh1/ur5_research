import mujoco
import gymnasium as gym

import ur5_env  # this runs gym.register for UR5-v0
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed 
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

import os
from datetime import datetime


def create_ur5_env():
    return gym.make("UR5-v0")

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode=None)  
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    wandb.init(
        project="ur5-ppo-training",
        config={
            "env_id": "UR5-v0",
            "algorithm": "PPO",
            "n_steps": 2048,
            "total_timesteps": 100_000,
            "num_cpu": 4,
            "log_std_init": -1.38, 
        }, 
        sync_tensorboard=True
    )

    env_id = "UR5-v0"
    num_cpu = 4

    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO(
        "MlpPolicy", 
        vec_env, 
        n_steps=2048, # how many timesteps do you need to do the right behavior 
        verbose=1,
        tensorboard_log="tensorboard_log", 
        policy_kwargs=dict(
            log_std_init=-1.38, 
        )
    )
    # stochastic policy hence you need to have a std parameter 
    # action is the mean 
    # std is used to play with that more / how spread out the sampling 
    # done in log space 
    model.learn(total_timesteps=100_000, callback=WandbCallback(verbose=2))

    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])

    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        # eval_env.render()

    wandb.finish()


if __name__ == "__main__":
    main()
