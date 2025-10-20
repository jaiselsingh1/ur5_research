
import gymnasium as gym
import ur5_push_env as ur5
import numpy as np
import time
import mppi_ur5 as mppi

env = ur5.ur5(render_mode="human", fix_orientation=False)  # 6-D actions
controller = mppi.MPPI(env)

for episode in range(5):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    if hasattr(controller, "U"):
        controller.U[:] = 0.0  

    for step in range(200):
        # Use the true MuJoCo state for planning (qpos|qvel), not an obs slice
        state = np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()], axis=0)

        action = controller.action(state)

        # Clip to env bounds (now 6-D)
        low, high = env.action_space.low, env.action_space.high
        if np.isfinite(low).all() and np.isfinite(high).all():
            action = np.clip(action, low, high)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        time.sleep(0.05)

env.close()


# import gymnasium as gym
# import ur5_push_env as ur5
# import numpy as np
# import time
# import mppi_ur5 as mppi 

# env = ur5.ur5(render_mode="human")
# controller = mppi.MPPI(env)

# for episode in range(5):
#     obs = env.reset()
    
#     # print(f"\n=== Episode {episode} ===")
#     # print("Tape at:", env.tape_roll.xpos)
#     # print("EE at:", env.ee_finger.xpos)
    
#     for step in range(200):
#         # Recalculate action each step
#         # direction = env.tape_roll.xpos - env.ee_finger.xpos
#         # distance = np.linalg.norm(direction)
        
#         # if distance < 0.05:
#         #     print(f"Reached tape at step {step}!")
#         #     break
        
#         # action = direction / distance
#         state = obs[: 26]
#         action = controller.action(state)
#         env.step(action)
        
#         # if step % 20 == 0:
#         #     print(f"  Step {step}: distance = {distance:.3f}m")
        
#         time.sleep(0.05)  # Slower
    
#     # print("Final EE:", env.ee_finger.xpos)
#     # input("Press Enter for next episode...")

# env.close()