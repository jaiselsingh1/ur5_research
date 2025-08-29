import mujoco
import gymnasium as gym
import ur5_env  # Make sure this imports your updated environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from datetime import datetime

def create_ur5_env():
    # Use your original environment ID
    env = gym.make("UR5-v0")
    return env

def setup_logging(env_name="ur5_scene"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_{env_name}_{timestamp}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"Logs: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    return log_dir

def main():
    log_dir = setup_logging()
    
    # Create training and evaluation environments
    train_env = make_vec_env(create_ur5_env, n_envs=1)
    eval_env = make_vec_env(create_ur5_env, n_envs=1)
    
    # You might need to adjust these hyperparameters for your specific task
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        # Optional: adjust network architecture if needed
        # policy_kwargs=dict(net_arch=[64, 64])  # Smaller network for simpler tasks
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print(f"Action space: {train_env.action_space}")
    print(f"Observation space: {train_env.observation_space}")
    
    # Train the model
    model.learn(
        total_timesteps=50000,  # Adjust based on your task complexity
        callback=eval_callback,
        tb_log_name="ppo_run"
    )
    
    # Save the final model
    model.save(f"{log_dir}/final_model")
    print(f"Training complete! Model saved to {log_dir}")

if __name__ == "__main__":
    main()
