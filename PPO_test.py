
import mujoco 
import gymnasium as gym 
import ur5_env 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from datetime import datetime

def create_ur5_env():
    env = gym.make("UR5-v0")
    env = TimeLimit(env, max_episode_steps=200)
    return env 

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_ur5_{timestamp}"
    