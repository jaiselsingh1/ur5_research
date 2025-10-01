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
        # control sequence over the horizon 
        self.U = torch.zeros((self.horizon, self.act_dim), dtype=torch.float32)
    
    def _sample_noise(self):
        return torch.randn((self.num_samples,self.horizon, self.act_dim)) * self.noise_sigma
    
    def _snapshot_state(self):
        state = mujoco.mj_getState(self.model, self.data, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state.copy()

    def _restore_state(self, data, state):
        state = mujoco.mj_setState(self.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, data)

    def _rollout(self, U_traj, curr_state):
        temp_data = mujoco.MjData(self.model)
        self._restore_state(temp_data, curr_state)

        for t in range(self.horizon):
            # U is the dimensions of self.act_dim over the horizon 
            temp_data.ctrl[:self.act_dim] = U_traj[t, :].cpu().numpy()
            mujoco.mj_step(self.model, temp_data)

            

    def _control(self):
        pass




        

