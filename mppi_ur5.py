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
            lambda_: float = 1.0,
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
        lam = self.lambda_
        
        states = np.repeat(state[None], self.num_samples, axis=0)
        states = np.concat([np.zeros((self.num_samples, 1)), states], axis=-1)

        noise = np.random.randn(self.num_samples, self.horizon, self.act_dim) * self.noise_sigma
        controls = self.U[None] + noise
        
        rollout_states, _ = rollout.rollout(self.model, self.data, states, controls, persistent_pool=True)
        # rollout state is (K, T, D)        

        # cost calculations
        ee_pos = rollout_states[:, :, 12:15]         # (samples, timesteps, state size)
        tape_pos = rollout_states[:, :, 19:22]
        target_pos = rollout_states[:, :, 22:25]

        # both are the same size
        ee_to_obj = np.linalg.norm(tape_pos - ee_pos, axis=-1)
        obj_to_tar = np.linalg.norm(target_pos - tape_pos, axis=-1)

        # instantaneous costs 
        q = 2.0 * ee_to_obj**2 + 10.0 * obj_to_tar**2 

        # second component of the S costs (aka control costs)
        # assuming no cross correlations in the noise
        ctrl_cost = lam * np.sum((noise / self.noise_sigma)**2, axis=-1)  # (K, T, A) -> (K, T)

        S = np.sum(q + ctrl_cost, axis=1) # (K, )

        # inline terminal cost
        x_t = rollout_states[:, -1, :]
        tape_t   = x_t[:, 19:22]
        target_t = x_t[:, 22:25]
        phi = 10.0 * np.linalg.norm(tape_t - target_t, axis=-1) # (K,)
        S += phi

        beta = np.min(S)
        weights_unnorm = np.exp(-(S - beta) / lam)                    # (K,)
        weights = weights_unnorm / (np.sum(weights_unnorm) + 1e-12)         # (K,)

        delta_U = np.einsum("k, kta -> ta", weights, noise)
        self.U += delta_U

        # apply first control
        u_0 = self.U[0].copy()  # (A,)
        self.U[:-1] = self.U[1:] # shift horizon 
        self.U[-1] = self.U[-2]

        return u_0

