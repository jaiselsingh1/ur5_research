import ur5_push_env
import numpy as np 
from sbx import PPO
import time

env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
obs, _ = env.reset()

 # load model
# model = PPO.load("./trained_models/ppo_ur5_p2b84fc1.zip")
model = PPO.load("./trained_models/ppo_ur5_aro8csko.zip")

for i in range(1000):

    if i % 100 == 0:
        obs, _ = env.reset()

    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    
    quat_error = np.linalg.norm(env.ee_finger.xquat - env.fixed_down_quat)

    
    # print(f"\nStep {i}:")
    # print(f"  Joint 0 (base): {env.data.qpos[0]:.3f}")
    # print(f"  Quat error: {quat_error:.4f}")
    # print(f"  Action: {action}")
    
    time.sleep(0.1)


# env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
# obs, _ = env.reset()

# print("EE orientation (quaternion):", env.ee_finger.xquat)
# print("Fixed downward quat:", env.fixed_down_quat)
# print("Are they the same?", np.allclose(env.ee_finger.xquat, env.fixed_down_quat, atol=0.1))

# input("Press enter to take a zero action...")
# env.step(np.zeros(3))


# import time 
# import ur5_push_env
# import numpy as np
# import cartesian_controller as cc 
# from sbx import PPO


# env = ur5_push_env.ur5(render_mode="human")
# obs, _ = env.reset()


# action = env.action_space.sample()
# action = np.zeros_like(action)
# action[0] = -0.1

# for i in range(1000):
#     obs, _, _, _, _= env.step(action)
#     if i % 100 == 0:
#         env.reset()
#     time.sleep(1/env.metadata["render_fps"])
#     env.render()

