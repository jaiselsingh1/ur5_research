import mujoco 
import ur5_push_env as ur5 
import cartesian_controller
import numpy as np 
import time 

env = ur5.ur5(render_mode="human", fix_orientation=True)

action = np.zeros(env.action_space.shape, dtype=np.float64)
action[0] = 1.0 
print(action.shape)

for i in range(10000):
    
    if i % 100 == 0:
        obs, _ = env.reset()

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.1)

        


    
