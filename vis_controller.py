
import gymnasium as gym
import ur5_push_env as ur5
import numpy as np
import time
import mppi_ur5 as mppi

env = ur5.ur5(render_mode="human", fix_orientation=True)  # 6-D actions
sim_env = ur5.ur5(fix_orientation=True)  # 6-D actions
controller = mppi.MPPI(sim_env)

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


# work in joint velocities instead since 6D 
# controller rollouts to get joint velocities 