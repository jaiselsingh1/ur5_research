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
        # when eval_env is a VecEnv, it does not expose env methods directly.
        # grab the first underlying env to access reward_dict() for the keys.
        base_env = config.eval_env.envs[0] if hasattr(config.eval_env, "envs") else config.eval_env
        base_env.reset()
        reward_dict = base_env.reward_dict()
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

                # flush logger so these show up in TB/W&B regularly
                self.logger.dump(self.num_timesteps)
        
        # periodic eval 
        if self.n_calls % self.config.eval_freq == 0:
            self._evaluate_agent()
            
        return True

    def _evaluate_agent(self):
        """evaluate the agent and then log the results"""
        episode_rewards = []
        episode_lengths = []
        episode_sucesses = []

        # vectorized env handling (also works for single raw env) 
        env = self.config.eval_env
        deterministic = self.config.deterministic

        # detect number of envs (VecEnv exposes .num_envs)
        num_envs = getattr(env, "num_envs", 1)

        # reset (VecEnv returns just obs; raw Gymnasium returns (obs, info))
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _info0 = reset_out
        else:
            obs = reset_out

        # running per-env accumulators
        ep_rets = np.zeros(num_envs, dtype=float)
        ep_lens = np.zeros(num_envs, dtype=int)
        comp_totals = [{k: 0.0 for k in self.reward_components.keys()} for _ in range(num_envs)]

        episodes_done = 0
        target_episodes = self.config.n_eval_episodes

        while episodes_done < target_episodes:
            # actions from policy
            actions, _ = self.model.predict(obs, deterministic=deterministic)

            step_out = env.step(actions)
            # VecEnv: (obs, rewards, dones, infos)
            # raw Gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 4:
                obs, rewards, dones, infos = step_out
                # ensure array shapes
                rewards = np.array(rewards).reshape(-1)
                dones = np.array(dones).reshape(-1)
                infos_list = list(infos) if isinstance(infos, (list, tuple)) else [infos]
            else:
                obs, reward, terminated, truncated, info = step_out
                rewards = np.array([reward], dtype=float)
                dones = np.array([bool(terminated or truncated)])
                infos_list = [info]

            # accumulate per-env stats
            ep_rets[:len(rewards)] += rewards
            ep_lens[:len(rewards)] += 1

            for i in range(len(infos_list)):
                info_i = infos_list[i]
                # accumulate component-wise rewards if present
                for k in comp_totals[i].keys():
                    if k in info_i:
                        comp_totals[i][k] += float(info_i[k])

                if dones[i]:
                    # record episode totals
                    episode_rewards.append(float(ep_rets[i]))
                    episode_lengths.append(int(ep_lens[i]))
                    episode_sucesses.append(bool(info_i.get("is_success", False)))
                    episodes_done += 1
                    if episodes_done >= target_episodes:
                        break
                    # reset accumulators for that env slot
                    ep_rets[i] = 0.0
                    ep_lens[i] = 0
                    comp_totals[i] = {k: 0.0 for k in self.reward_components.keys()}

        # aggregate results
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        success_rate = float(np.mean(episode_sucesses)) if episode_sucesses else 0.0

        # log
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.dump(self.num_timesteps)
