
import mujoco 
import gymnasium as gym 
import ur5_env 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from datetime import datetime

def create_ur5_env():
    env = gym.make("UR5-v0")
    # env = TimeLimit(env, max_episode_steps=200)
    return env 

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_ur5_{timestamp}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"Logs: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    return log_dir

def main():
    log_dir = setup_logging()

    train_env = make_vec_env(create_ur5_env, n_envs=1)
    eval_env = make_vec_env(create_ur5_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,      # Standard PPO batch size
        batch_size=64,
        n_epochs=10,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=1000,           # Evaluate every 1000 steps
        n_eval_episodes=5,        # Use 5 episodes for evaluation
        deterministic=True,
        render=False
    )

    print("Starting training...")
    
    # Train the model
    model.learn(
        total_timesteps=50000,    # Longer training for robotics
        callback=eval_callback,
        tb_log_name="ppo_run"
    )

if __name__ == "__main__":
    main()
