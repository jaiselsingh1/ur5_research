import time 
import ur5_push_env
import numpy as np
import cartesian_controller as cc 
from sbx import PPO


env = ur5_push_env.ur5(render_mode="human")
obs, _ = env.reset()

action = env.action_space.sample()
action = np.zeros_like(action)
action[1] = 1.0

for i in range(1000):
    obs, _, _, _, _= env.step(action)
    if i % 100 == 0:
        env.reset()
    time.sleep(1/env.metadata["render_fps"])
    env.render()

