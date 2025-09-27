import mujoco
import gymnasium as gym

import ur5_env  # this runs gym.register for UR5-v0
import ur5_push_env  # for pushing task

from sbx import PPO 
import typing 
# from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed 
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

import os
from datetime import datetime


class PPOConfig(typing.NamedTuple):
    # Environment settings
    env_id: str = "UR5-v1"
    num_cpu: int = 8
    
    # PPO hyperparameters
    learning_rate: float = 1e-4
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 10
    total_timesteps: int = 50_000_000
    net_arch: dict = dict(pi=[128, 128], vf=[256, 256])
    
    # Policy settings
    log_std_init: float = -1.0
    
    # Evaluation settings
    eval_freq: int = 10_000
    
    # Logging settings
    tensorboard_log: str = "tensorboard_log"
    save_path: str = "./logs/"
    model_save_path: str = "./trained_models"
    # norm_path: str = "./vecnormalize.pkl"   
    
    # Wandb settings
    wandb_project: str = "ur5-ppo-training"
    gradient_save_freq: int = 100
    
    # Evaluation settings
    eval_episodes: int = 10
    eval_max_steps: int = 500


def create_ur5_env():
    return gym.make("UR5-v1")


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode=None)  
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def setup_wandb(config: PPOConfig):
    """Initialize wandb with config parameters"""
    wandb_config = config._asdict()
    wandb_config["algorithm"] = "PPO"  
    
    return wandb.init(
        project=config.wandb_project,
        config=wandb_config,
        sync_tensorboard=True 
    )


def create_model(config: PPOConfig, vec_env):
    """Create PPO model with config parameters"""
    return PPO(
        "MlpPolicy", 
        vec_env, 
        # ent_coef=0.01, 
        learning_rate=config.learning_rate,
        batch_size=config.batch_size, 
        n_epochs=config.n_epochs, # do not want to overfit the very small set of data that I am using 
        n_steps=config.n_steps,   # how many timesteps do you need to do within the environment for "right behavior" to do policy update
        verbose=1,
        tensorboard_log=config.tensorboard_log, 
        policy_kwargs=dict(
            log_std_init=config.log_std_init, 
            net_arch=config.net_arch
        )
    )
    # stochastic policy hence you need to have a std parameter 
    # action is the mean 
    # std is used to play with that more / how spread out the sampling 
    # done in log space 


def create_callbacks(config: PPOConfig):
    """Create evaluation and wandb callbacks"""
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(config.env_id))])
    # eval_env = VecNormalize(
    #     eval_env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     gamma=0.99,
    #     training=False  #  do not update stats during evaluation
    # )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.save_path,
        log_path=config.save_path,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=config.gradient_save_freq,  
        log="all",               # log metrics, gradients, model checkpoints
        verbose=2
    )
    
    return CallbackList([eval_callback, wandb_callback])


def evaluate_model(model, config: PPOConfig):
    """Evaluate trained model with human rendering"""
    # raw env with rendering
    eval_env_human = DummyVecEnv([lambda: gym.make(config.env_id, render_mode="human")])

    # load normalization stats so evaluation matches training
    # eval_env_human = VecNormalize.load(config.norm_path, eval_env_human)
    # eval_env_human.training = False         # do not update stats
    # eval_env_human.norm_reward = False      # keep rewards unnormalized (human interpretable)
    
    for _ in range(config.eval_episodes):
        obs = eval_env_human.reset()
        for _ in range(config.eval_max_steps):
            action, _ = model.predict(obs)
            obs, rewards, dones, info = eval_env_human.step(action)
            eval_env_human.render()  # keep human rendering


def main():
    config = PPOConfig()
    run = setup_wandb(config)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.num_cpu)])
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = create_model(config, vec_env)
    callbacks = create_callbacks(config)

    # Train
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)

    # Save model + normalization stats
    model.save(f"{config.model_save_path}/ppo_ur5_{run.id}.zip")

    # Evaluate model
    evaluate_model(model, config)
    wandb.finish()


if __name__ == "__main__":
    main()




# import mujoco
# import gymnasium as gym

# import ur5_env  # this runs gym.register for UR5-v0
# import ur5_push_env # for pushing task

# from sbx import PPO 
# import typing 
# # from stable_baselines3 import PPO

# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.callbacks import EvalCallback, CallbackList
# from stable_baselines3.common.utils import set_random_seed 
# from stable_baselines3.common.monitor import Monitor

# import wandb
# from wandb.integration.sb3 import WandbCallback

# import os
# from datetime import datetime

# class PPOConfig(typing.NamedTuple):
#     env_id : str = "UR5-v1"
#     n_steps: int = 2048
#     total_timesteps: int = 2_000_000
#     num_cpu: int = 4
#     log_std_init: float = -0.92
#     n_epochs: int = 10


# def create_ur5_env():
#     return gym.make("UR5-v1")

# def make_env(env_id: str, rank: int, seed: int = 0):
#     def _init():
#         env = gym.make(env_id, render_mode=None)  
#         env = Monitor(env)
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init


# def main():
#     wandb.init(
#         project="ur5-ppo-training",
#         config={
#             "env_id": "UR5-v1",
#             "algorithm": "PPO",
#             "n_steps": 2048,
#             "total_timesteps": 2_000_000,
#             "num_cpu": 4,
#             "log_std_init": -0.92, 
#         }, 
#         sync_tensorboard=True 
#     )

#     env_id = "UR5-v1"
#     num_cpu = 4

#     vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

#     model = PPO(
#         "MlpPolicy", 
#         vec_env, 
#         # ent_coef=0.01, 
#         learning_rate=1e-4,
#         batch_size=64, 
#         n_epochs=10, # do not want to overfit the very small set of data that I am using 
#         n_steps=2048, # how many timesteps do you need to do within the environment for "right behavior" to do policy update
#         verbose=1,
#         tensorboard_log="tensorboard_log", 
#         policy_kwargs=dict(
#             log_std_init=-0.92, 
#         )
#     )
#     # stochastic policy hence you need to have a std parameter 
#     # action is the mean 
#     # std is used to play with that more / how spread out the sampling 
#     # done in log space 

#     eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
 
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path="./logs/",
#         log_path="./logs/",
#         eval_freq=10_000,
#         deterministic=True,
#         render=False
#     )

#     callbacks = CallbackList([
#         eval_callback,
#         WandbCallback(
#             gradient_save_freq=100,  
#             log="all",               # log metrics, gradients, model checkpoints
#             verbose=2
#         )
#     ])

#     model.learn(total_timesteps=2_000_000, callback=callbacks) #callback=WandbCallback(verbose=2))
#     model.save("./trained_models.pt")

#     eval_env_human = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])

#     for _ in range(10):
#         obs = eval_env_human.reset()
#         for _ in range(1000):
#             action, _ = model.predict(obs)
#             obs, rewards, dones, info = eval_env_human.step(action)
#             # eval_env_human.render()
#     wandb.finish()


# if __name__ == "__main__":
#     main()
