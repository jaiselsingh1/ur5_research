
import mujoco
import numpy as np
import ur5_env 

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

# MLP Policy since feature vector is being used and not images 
env = ur5_env.ur5()
model = PPO(MlpPolicy, env, verbose=0)

def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """
    Evaluate a RL agent for num_episodes
    :param model: RL agent
    :param env: the gym environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    
    for _ in range(num_episodes):  # Fixed: underscore for unused variable
        episode_rewards = []
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)  # Fixed: vec_env typo
            episode_rewards.append(reward)
        
        all_episode_rewards.append(sum(episode_rewards))
    
    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")
    return mean_episode_reward

# Random Agent, before training 
mean_reward_before_train = evaluate(model, num_episodes = 100, deterministic = True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

#train the agent for 10000 steps 
model.learn(total_timesteps = 10000)

#evaluate the trained agent 
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Recording Code which I copied from a tutorial 

# Set up fake display; otherwise rendering will fail
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

import base64
from pathlib import Path

from IPython import display as ipythondisplay


def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


record_video("UR5-v0", model, video_length=500, prefix="ppo-ur5")
show_videos("videos", prefix="ppo")

model = PPO('MlpPolicy', env, verbose=1).learn(1000)












