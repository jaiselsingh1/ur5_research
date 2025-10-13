import time
import ur5_push_env
import numpy as np
from scipy.spatial.transform import Rotation
from sbx import PPO

env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
obs, _ = env.reset()

#  # load model
model = PPO.load("./trained_models/ppo_ur5_4mx4eqmh.zip")
# action = np.zeros(6)

for i in range(10000):

    if i % 100 == 0:
        obs, _ = env.reset()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, reward_dict = env.step(action)
    print(reward_dict)

    time.sleep(0.1)



