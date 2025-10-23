import time
import ur5_push_env
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from sbx import PPO

env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
obs, _ = env.reset()

# load model
model = PPO.load("./trained_models/ppo_ur5_z4znoylk.zip")
# action = np.zeros(6)

for i in range(10000):
    # import ipdb 
    # ipdb.set_trace()
    
    if i % 100 == 0:
        obs, _ = env.reset()
        # env.data.qpos[9:] = np.array([0.707, 0.0, 0.707, 0.0])

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, reward_dict = env.step(action)
    print(env.reward_dict()["contact"])

    time.sleep(0.1)



