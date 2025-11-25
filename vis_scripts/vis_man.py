import mujoco 
import sys 
sys.path.append("..")
import ur5_push_env as ur5 
import cartesian_controller
import numpy as np 
import time 

env = ur5.ur5(render_mode="human", fix_orientation=True)

action = np.zeros(env.action_space.shape, dtype=np.float64)
action[0] = 1.0 
action[1] = 1.0

# 0.502 0.0 -0.18
# size 0.3 0.6 0.03

for i in range(10000):

    if i % 100 == 0:
        obs, _ = env.reset()

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.1)

        


    
