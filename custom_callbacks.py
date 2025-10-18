import typing 
import mujoco 
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np 
import os


class ur5_callback_config(typing.NamedTuple):
        # named tuple is a special type of class that has it's own init so you don't define it for you 
        eval_env: gym.Env
        n_eval_episodes: int
        log_freq: int
        eval_freq: int 
        deterministic: bool = True 
        verbose: int = -1
        best_model_save_path: str = ""
        render: bool = False 

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
        """run evaluation episodes on the agent and then log the results"""
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        ep_component_totals = []
        # record the component totals per episode (sum over steps and then report per step average)

        env = self.config.eval_env 
        deterministic = self.config.deterministic 

        for _ in range(self.config.n_eval_episodes):
            obs, info = env.reset()
            done = False 
            ep_ret = 0.0 # returns the rewards of the eval episode
            ep_len = 0
            last_info = {}

            comp_totals = {k: 0.0 for k in self.reward_components.keys()}

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                ep_ret += float(reward)
                ep_len += 1
                last_info = info

                for k in comp_totals.keys():
                    if k in info:
                        # info is from the step function that you're returning from 
                        comp_totals[k] += float(info[k])

                if self.config.render:
                    env.render()

            # totals (append once per episode)
            episode_rewards.append(ep_ret)
            episode_lengths.append(ep_len)
            episode_successes.append(bool(last_info.get("is_success", False)))
            ep_component_totals.append((comp_totals, ep_len))

        # calculate means from all episodes 
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
        
        # Component per-step averages (sum over episode / ep_len, then mean across episodes)
        comp_means = {}
        for k in self.reward_components.keys():
            per_ep = []
            for totals, L in ep_component_totals:
                if L > 0:
                    per_ep.append(totals[k] / L)
            comp_means[k] = float(np.mean(per_ep)) if per_ep else 0.0

        # Log to SB3's logger
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/success_rate", success_rate)
        for k, v in comp_means.items():
            self.logger.record(f"eval_reward_components/{k}", v)

        # Dump at this timestep (so it appears in TensorBoard/W&B immediately)
        self.logger.dump(self.num_timesteps)

        # Save the best model so far
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.config.best_model_save_path:
                os.makedirs(self.config.best_model_save_path, exist_ok=True)
                save_path = os.path.join(self.config.best_model_save_path, "best_model")
                self.model.save(save_path)
