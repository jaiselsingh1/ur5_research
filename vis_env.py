import time 
import ur5_push_env
env = ur5_push_env.ur5(render_mode="human")
import numpy as np
observation, info = env.reset()


i = 0 
while True:
    i += 1

    action = np.zeros(6)
    action[4] = 1.0
    if i % 200 == 0:
        env.reset()

        print("\n \n \n")

    env.step(action)
    env.render()
    # time.sleep(0.1)