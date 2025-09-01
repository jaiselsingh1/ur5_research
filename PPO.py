import mujoco
import gymnasium as gym
import ur5_env  # Make sure this imports your updated environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed 


import os
from datetime import datetime

def create_ur5_env():
    # Use your original environment ID
    env = gym.make("UR5-v0")
    return env

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed = seed + rank)
        return env
    set_random_seed(seed)
    return _init


def setup_logging(env_name="ur5_scene"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_{env_name}_{timestamp}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"Logs: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    return log_dir

def main():
    # log_dir = setup_logging()
    
    env_id = "UR5-v0"
    num_cpu = 4 # number of processes to use 
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=1000)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

if __name__ == "__main__":
    main()
