import torch
import mujoco
from mujoco import rollout
from gymnasium import spaces 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from jaxtyping import Float
import numpy as np
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
        
        env.reset()
        self.model = env.model 
        self.data = env.data

        self.act_dim = 6
        #env.action_space.shape[0]
        # control sequence over the horizon 
        self.U = np.zeros((self.horizon, self.act_dim))

    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        
        states = np.repeat(state[None], self.num_samples, axis=0)
        states = np.concat([np.zeros((self.num_samples, 1)), states], axis=-1)

        noise = np.random.randn(self.num_samples, self.horizon, self.act_dim) * self.noise_sigma
        controls = self.U[None] + noise
        
        rollout_states, _ = rollout.rollout(self.model, self.data, states, controls, persistent_pool=True)
        
        # cost calculations
        ee_pos = rollout_states[:, :, 12:15]         # (samples, timesteps, state size)
        tape_pos = rollout_states[:, :, 19:22]
        target_pos = rollout_states[:, :, 22:25]

        # both are the same size
        ee_to_obj = np.linalg.norm(tape_pos - ee_pos)
        obj_to_tar = np.linalg.norm(target_pos - tape_pos)

        # instantaneous costs 
        q = 2.0 * ee_to_obj**2 + 10.0 * obj_to_tar**2 

        # regularization term + control costs 
        # sigma_inv = 




        return rollout_states 

