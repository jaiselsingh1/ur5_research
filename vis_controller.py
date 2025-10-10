import gymnasium as gym
import ur5_push_env as ur5
import numpy as np
import time

env = ur5.ur5(render_mode="human")

for episode in range(5):
    obs = env.reset()
    
    print(f"\n=== Episode {episode} ===")
    print("Tape at:", env.tape_roll.xpos)
    print("EE at:", env.ee_finger.xpos)
    
    for step in range(200):
        # Recalculate action each step
        direction = env.tape_roll.xpos - env.ee_finger.xpos
        distance = np.linalg.norm(direction)
        
        if distance < 0.05:
            print(f"Reached tape at step {step}!")
            break
        
        action = direction / distance
        env.step(action)
        
        if step % 20 == 0:
            print(f"  Step {step}: distance = {distance:.3f}m")
        
        time.sleep(0.05)  # Slower
    
    print("Final EE:", env.ee_finger.xpos)
    input("Press Enter for next episode...")

env.close()