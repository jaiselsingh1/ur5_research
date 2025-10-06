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
        
        return rollout_states

    
    # def _sample_noise(self):
    #     return torch.randn((self.num_samples,self.horizon, self.act_dim)) * self.noise_sigma
    
    # def _snapshot_state(self):
    #     state = mujoco.mj_getState(self.model, self.data, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    #     return state.copy()

    # def _restore_state(self, data, state):
    #     state = mujoco.mj_setState(self.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    #     mujoco.mj_forward(self.model, data)

    # def _rollout(self, U_traj, curr_state):
    #     temp_data = mujoco.MjData(self.model)
    #     self._restore_state(temp_data, curr_state)

    #     states, _ = mujoco.rollout.rollout(self.model, temp_data, curr_state, U_traj, persistent_pool=True)

    #     # for t in range(self.horizon):
    #     #     # U is the dimensions of self.act_dim over the horizon 
    #     #     temp_data.ctrl[:self.act_dim] = U_traj[t, :].cpu().numpy()
    #     #     mujoco.mj_step(self.model, temp_data)     

    # def _control(self):
    #     pass
