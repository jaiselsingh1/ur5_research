import time 
import ur5_push_env
# env = ur5_push_env.ur5(render_mode="rgb_array")
import numpy as np
import cartesian_controller as cc 
from sbx import PPO
import imageio 

env = ur5_push_env.ur5(render_mode="rgb_array")
policy = PPO.load("./logs/best_model.zip")
writer = imageio.get_writer("ur5_policy.mp4", fps=30, codec="libx264")

num_rollouts = 10
steps_per_rollout = 500

for rollout in range(num_rollouts):
    obs, info = env.reset()
    for step in range(steps_per_rollout):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        # Save every frame
        frame = env.render()
        writer.append_data(frame)

        # If env ends early, break and reset next rollout
        if terminated or truncated:
            break

writer.close()



# obs, info = env.reset()
# i = 0 
# observations = []
# policy = PPO.load("./logs/best_model.zip")
# for i in range(5000):
#     # i += 1
#     action = env.action_space.sample()
#     action, _ = policy.predict(obs, deterministic=False)

#     if i % 500 == 0:
#         env.reset()
#         print("\n ")
#         # target_pos = np.array([np.random.uniform(0.3, 0.6),  # within table X
#         #               np.random.uniform(-0.1, 0.5),     # within table Y  
#         #               np.random.uniform(-0.1475, -0.05) # above surface
#         #               ])
#         # target_quat = np.array([1, 0, 0.0, 0]) # xyzw 
#         # max_iters = 1000
#         # tol = 1e-5
#         # for _ in range(max_iters):
#         #     x_command = cc.Command(*target_pos, *target_quat) # takes an iterable and passes it as arguments in the order
#         #     qvel = np.clip(env.cartesian_controller.cartesian_command(env.data.qpos, x_command), -3.15, 3.15)
#         #     env.do_simulation(qvel, 1) # the controller is a velocity controller so you can just pass in qvel

#         #     # error = np.linalg.norm(env.ee_finger.xpos - target_pos)
#         #     # if error < tol:
#         #     #     break 
#         # # print(error)
#         # print(env.data.qpos)
#         # print(env.ee_finger.xquat)

#     obs, _, _, _, _= env.step(action)
#     #observations.append(obs)
#     env.render()
