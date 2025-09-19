import time 
import ur5_push_env
env = ur5_push_env.ur5(render_mode="human")
import numpy as np
import cartesian_controller as cc 
observation, info = env.reset()


i = 0 
observations = []
for i in range(1000):
    # i += 1
    action = env.action_space.sample()
    action = np.zeros_like(action)
    if i % 50 == 0:
        env.reset()

        print("\n ")
        # target_pos = np.array([np.random.uniform(0.3, 0.6),  # within table X
        #               np.random.uniform(-0.1, 0.5),     # within table Y  
        #               np.random.uniform(-0.1475, -0.05) # above surface
        #               ])
        # target_quat = np.array([1, 0, 0.0, 0]) # xyzw 
        # max_iters = 1000
        # tol = 1e-5
        # for _ in range(max_iters):
        #     x_command = cc.Command(*target_pos, *target_quat) # takes an iterable and passes it as arguments in the order
        #     qvel = np.clip(env.cartesian_controller.cartesian_command(env.data.qpos, x_command), -3.15, 3.15)
        #     env.do_simulation(qvel, 1) # the controller is a velocity controller so you can just pass in qvel

        #     # error = np.linalg.norm(env.ee_finger.xpos - target_pos)
        #     # if error < tol:
        #     #     break 
        # # print(error)
        # print(env.data.qpos)
        # print(env.ee_finger.xquat)

    obs, _, _, _, _= env.step(action)

    #observations.append(obs)
    env.render()
    # time.sleep(0.1)
# all_obs = np.stack(observations)
# print(all_obs.mean(axis=0))
# print(all_obs.std(axis=0))