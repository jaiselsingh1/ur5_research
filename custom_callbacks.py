import typing 
import mujoco 
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np 


class ur5_callback_config(typing.NamedTuple):
        # named tuple is a special type of class that has it's own init so you don't define it for you 
        eval_env: gym.Env
        n_eval_episodes: int
        log_freq: int
        eval_freq: int 
        deterministic: bool = True 
        verbose: int = -1
        best_model_save_path: str = ""
        deterministic: bool = True
        render: bool = False 
        verbose: int = 1 

class ur5_callback(BaseCallback):
    def __init__(self, config: ur5_callback_config):
        super().__init__(config.verbose)
        self.config = config 
        self.best_mean_reward = -np.inf 

        # reward_scales = self.config.eval_env.reward_scales()
        config.eval_env.reset()
        reward_dict = config.eval_env.reward_dict()
        # you need a list to store the rewards over time though 
        self.reward_components = {key: [] for key in reward_dict.keys()}

    def _on_step(self) -> bool:
        """called at each step of the training process"""
        if self.n_calls % self.config.log_freq == 0:  
            if 'infos' in self.locals:  # self.locals is a dict that contains the current training steps that is maintained by SB3
                infos = self.locals['infos']
    
        for key in self.reward_components.keys():
                    values = [info.get(key, 0.0) for info in infos if key in info]
                    if values:
                        mean_value = np.mean(values)
                        self.reward_components[key].append(mean_value)
                        self.logger.record(f"reward_components/{key}", mean_value)
        
        # periodic eval 
        if self.n_calls % self.config.eval_freq == 0:
            self._evaluate_agent()
            
        return True

    def _evaluate_agent(self):
        """evaluate the agent and then log the results"""
        episode_rewards = []
        episode_lengths = []
        episode_sucesses = []

        



    




            
