import torch
import mujoco
from gymnasium import spaces 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import typing

class MPPI:
    def __init__(
            self, 
            env: MujocoEnv, 
            num_samples: int = 256, # number of samples 
            horizon: int = 10, # number of time steps 
            noise_sigma: float = 0.2, 
            phi: float = None, 
            q: float = None, 
            lambda_: float = None,
    ):
        
        self.num_samples = num_samples
        self.horizon = horizon 
        self.noise_sigma = noise_sigma
        self.phi = phi 
        self.q = q
        self.lambda_= lambda_
        
        self.model = env.model 
        self.data = env.data

        self.act_dim = env.action_space.shape[0]
        
        

