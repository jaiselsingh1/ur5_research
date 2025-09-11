import time 
import ur5_push_env
env = ur5_push_env.ur5(render_mode="human")
import numpy as np
env.reset()


i = 0 

while True:
    i += 1
    print(f"{env.ee_finger.xquat} original quat")
    print(f"{env.ee_finger.xpos} original position")
    print(f"{env.ee_target_pos} target initial xpos")
    print(f"{env.ee_target_quat} target initial xquat")
    action = np.array([0,0,0,0,0,0])

    if i % 20 == 0:
        env.reset()
        print("\n \n \n")

    env.step(action)
    env.render()
    time.sleep(0.1)