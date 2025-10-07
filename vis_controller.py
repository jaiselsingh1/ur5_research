import mppi_ur5 as mppi 
import ur5_push_env as ur5 
import numpy as np

rollout_env = ur5.ur5(render_mode="rgb_array")
vis_env = ur5.ur5(render_mode="human")
planner = mppi.MPPI(env = rollout_env)


print(rollout_env.data.qpos.shape, rollout_env.data.qvel.shape)
states = planner.action(np.concat([rollout_env.data.qpos, rollout_env.data.qvel]))

print(states)





