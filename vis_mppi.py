import time
import ur5_push_env
import numpy as np
from scipy.spatial.transform import Rotation
from sbx import PPO

env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
model = PPO.load("./trained_models/ppo_ur5_n6t3rvdr.zip")

print("="*60)
print("POLICY BEHAVIOR ANALYSIS")
print("="*60)

for episode in range(3):
    print(f"\n{'='*60}")
    print(f"Episode {episode + 1}")
    print(f"{'='*60}")
    
    obs, _ = env.reset()
    
    # Track metrics over episode
    base_positions = []
    quat_errors = []
    ee_to_tape_dists = []
    rewards = []
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        
        # Track metrics
        base_positions.append(env.data.qpos[0])
        q_cur = Rotation.from_quat(env.ee_finger.xquat, scalar_first=True)
        ang_err = (env.q_des * q_cur.inv()).magnitude()
        quat_errors.append(ang_err)
        ee_to_tape_dists.append(np.linalg.norm(env.ee_finger.xpos - env.tape_roll.xpos))
        rewards.append(reward)
        
        time.sleep(1/env.metadata["render_fps"])
        
        if term or trunc:
            print(f"Episode ended at step {step}")
            break
    
    # Analyze behavior
    base_positions = np.array(base_positions)
    quat_errors = np.array(quat_errors)
    ee_to_tape_dists = np.array(ee_to_tape_dists)
    rewards = np.array(rewards)
    
    print(f"\nBase rotation (joint 0):")
    print(f"  Range: {base_positions.min():.3f} to {base_positions.max():.3f} rad")
    print(f"  Total rotation: {abs(base_positions.max() - base_positions.min()):.3f} rad ({np.degrees(abs(base_positions.max() - base_positions.min())):.1f}°)")
    print(f"  Mean: {base_positions.mean():.3f}, Std: {base_positions.std():.3f}")
    
    print(f"\nOrientation error:")
    print(f"  Range: {quat_errors.min():.3f} to {quat_errors.max():.3f} rad ({np.degrees(quat_errors.min()):.1f}° to {np.degrees(quat_errors.max()):.1f}°)")
    print(f"  Mean: {quat_errors.mean():.3f} rad ({np.degrees(quat_errors.mean()):.1f}°)")
    
    print(f"\nEE to tape distance:")
    print(f"  Start: {ee_to_tape_dists[0]:.3f}m")
    print(f"  End: {ee_to_tape_dists[-1]:.3f}m")
    print(f"  Min reached: {ee_to_tape_dists.min():.3f}m")
    
    print(f"\nRewards:")
    print(f"  Mean: {rewards.mean():.2f}")
    print(f"  Total: {rewards.sum():.2f}")
    
    print(f"\nDid it reach the tape? {'YES' if ee_to_tape_dists.min() < 0.05 else 'NO'}")

env.close()



# import ur5_push_env
# import numpy as np 
# from sbx import PPO
# import time

# env = ur5_push_env.ur5(render_mode="human", fix_orientation=True)
# obs, _ = env.reset()

#  # load model
# # model = PPO.load("./trained_models/ppo_ur5_p2b84fc1.zip")
# model = PPO.load("./trained_models/ppo_ur5_n6t3rvdr.zip")

# for i in range(1000):

#     if i % 100 == 0:
#         obs, _ = env.reset()

#     action, _ = model.predict(obs)
#     obs, reward, done, truncated, _ = env.step(action)
    
#     quat_error = np.linalg.norm(env.ee_finger.xquat - env.fixed_down_quat)
#     print(quat_error)
    
    
#     time.sleep(0.1)



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

