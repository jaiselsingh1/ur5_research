import typing 
import mujoco 
import gymnasium as gym


class ur5_Callback(typing.NamedTuple):
            # named tuple is a special type of class that has it's own init so you don't define it for you 
            eval_env: gym.Env
            n_eval_episodes: int
            eval_freq: int
            deterministic: bool = True 
            verbose: int = -1
            best_model_save_path: str = ""

