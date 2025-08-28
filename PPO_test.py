import mujoco
import numpy as np
import ur5_env   # make sure your UR5 env is registered as "UR5-v0"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import time

def quick_test_env():
    """Test if environment can be created and stepped through"""
    print("Testing environment creation...")
    try:
        env = gym.make("UR5-v0")
        obs, info = env.reset()
        print(f"✓ Environment created. Obs shape: {obs.shape}")
        
        # Test a few random steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                obs, info = env.reset()
                
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def quick_evaluate(model: BaseAlgorithm, num_episodes: int = 3, deterministic: bool = True) -> float:
    """Quick evaluation with minimal episodes and timeout protection"""
    print(f"Starting evaluation with {num_episodes} episodes...")
    
    try:
        # Use the built-in evaluation directly - it's more robust
        mean_reward, std_reward = evaluate_policy(
            model, 
            model.get_env(), 
            n_eval_episodes=num_episodes, 
            warn=False,
            deterministic=deterministic
        )
        print(f"Quick eval -> Mean reward: {mean_reward:.2f} ± {std_reward:.2f} over {num_episodes} episodes")
        return mean_reward
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Trying manual evaluation with timeout...")
        
        # Fallback: manual evaluation with step limit
        vec_env = model.get_env()
        obs = vec_env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        total_reward = 0
        steps_taken = 0
        max_steps_per_episode = 200  # Prevent infinite episodes
        episodes_completed = 0

        while episodes_completed < num_episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            result = vec_env.step(action)

            if len(result) == 4:  # VecEnv
                obs, reward, done, info = result
            elif len(result) == 5:  # Raw gym env
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step() return format: {len(result)} elements")

            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps_taken += 1
            
            # Force episode end if too long
            if steps_taken >= max_steps_per_episode:
                print(f"  Episode {episodes_completed + 1} forced to end at {steps_taken} steps")
                done = True

            if done:
                episodes_completed += 1
                print(f"  Episode {episodes_completed} completed in {steps_taken} steps")
                steps_taken = 0
                if episodes_completed < num_episodes:
                    obs = vec_env.reset()
                    if isinstance(obs, tuple):
                        obs, _ = obs

        mean_reward = total_reward / num_episodes
        print(f"Manual eval -> Mean reward: {mean_reward:.2f} over {num_episodes} episodes")
        return mean_reward

def main():
    print("=== Quick PPO Test ===")
    
    # Step 1: Test environment
    if not quick_test_env():
        return
    
    print("\n=== Setting up PPO ===")
    env = make_vec_env("UR5-v0", n_envs=1)  # Single environment for simplicity
    
    # Create model with progress tracking
    model = PPO(MlpPolicy, env, verbose=1, 
               learning_rate=3e-4,
               n_steps=64,        # Smaller batch for faster iterations
               batch_size=32,     # Smaller batches
               n_epochs=4)        # Fewer epochs per update
    
    print("\n=== Pre-training Evaluation ===")
    # Skip pre-training eval for now to test training
    print("Skipping pre-training evaluation to test training directly...")
    pre_reward = -999  # Placeholder
    
    print("\n=== Training (with progress) ===")
    start_time = time.time()
    
    # Train with callbacks to see progress
    model.learn(total_timesteps=2000,  # Much smaller for quick test
                progress_bar=True)     # Show progress bar
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f} seconds")
    
    print("\n=== Post-training Evaluation ===")
    post_reward = quick_evaluate(model, num_episodes=5)
    
    print("\n=== Results Summary ===")
    print(f"Pre-training reward:  {pre_reward:.2f}")
    print(f"Post-training reward: {post_reward:.2f}")
    print(f"Improvement:          {post_reward - pre_reward:.2f}")
    print(f"Training time:        {train_time:.1f}s")
    
    # Optional: Save model for later use
    model.save("ur5_ppo_quick_test")
    print("Model saved as 'ur5_ppo_quick_test.zip'")

def test_with_callback():
    """Alternative version with evaluation callback during training"""
    print("\n=== PPO with Training Callback ===")
    
    env = make_vec_env("UR5-v0", n_envs=1)
    eval_env = make_vec_env("UR5-v0", n_envs=1)
    
    model = PPO(MlpPolicy, env, verbose=0)  # Less verbose since callback will report
    
    # Callback evaluates every 500 steps during training
    eval_callback = EvalCallback(eval_env, 
                                best_model_save_path='./best_model/',
                                log_path='./logs/', 
                                eval_freq=500,
                                n_eval_episodes=3,
                                deterministic=True, 
                                render=False)
    
    print("Training with evaluation callback...")
    model.learn(total_timesteps=2000, callback=eval_callback)
    print("Training with callback completed!")

# Run the test directly
main()
test_with_callback()