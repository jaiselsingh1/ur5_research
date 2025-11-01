from dataclasses import dataclass, asdict 
from pathlib import Path  
import numpy as np 
import yaml 
import time 

@dataclass 
class TrajMeta:
    seed: int 
    workspace_bounds: dict # {"x":[min,max], "y":[min,max], "z":[min,max]}
    target_position: list  # [x,y,z] (not visible in video)
    control_hz: float # 10 Hz 
    env_id: str 
    policy_tag: str 
    reward_version: str # internal tag 
    sim_timestep: float # model.opt.timestep 
    frame_skip: float 
    # filled at end 
    success: bool
    final_obj_to_target: float
    steps: int    
    # optional
    notes: str = ""

class TrajectoryWriter:
    def __init__(self, root_dir: str, dataset_name: str, start_idx: int = 1):
        self.root = Path(root_dir) / dataset_name 
        self.root.mkdir(parents = True, exist_ok = True)
        self.next_idx = start_idx 
        self.reset_buffers()

    def reset_buffers(self):
        self.ee_pos_states = []
        self.ee_pos_actions = []         # cartesian actions/commands
        self.obj_pos_states = []
        self.joint_value_states = []
        self.joint_vel_commands = []
        self.ee_pos_commands = []        # same as actions for clarity (matches sample)
        self.rewards = []                # optional, not saved if you don’t want
        self.timestamps = []             # not sure if this is needed 

    def start_traj(self):
        self.reset_buffers()
        self.t0 = time.time()

    def record_step(self, *, env, action, dq_cmd, reward):
        ee = env.ee_finger.xpos.copy() 
        obj = env.tape_roll.xpos.copy()
        q  = env.data.qpos[:6].copy()

        self.ee_pos_states.append(ee)
        self.ee_pos_actions.append(np.asarray(action, dtype=np.float32))  # xyz command (normalized * max_delta_pos)
        self.obj_pos_states.append(obj)
        self.joint_value_states.append(q)
        self.joint_vel_commands.append(np.asarray(dq_cmd[:6], dtype=np.float32))
        self.ee_pos_commands.append(ee)   # could switch to store commanded EE pos; here we store current EE pos
        self.rewards.append(float(reward))
        self.timestamps.append(time.time() - self.t0)

    def _npz(self, arr_list):
        # variable length time axes allowed (T, D)
        return np.asarray(arr_list, dtype=np.float32)
    
    def finish_traj(self, meta: TrajMeta):
        traj_dir = self.root / f"traj_{self.next_idx:06d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # .npz files (names match the samples)
        np.savez_compressed((traj_dir / "ee_pos_states.npz"), self._npz(self.ee_pos_states))
        np.savez_compressed(traj_dir / "ee_pos_actions.npz",     self._npz(self.ee_pos_actions))
        np.savez_compressed(traj_dir / "obj_pos_states.npz",     self._npz(self.obj_pos_states))
        np.savez_compressed(traj_dir / "joint_value_states.npz", self._npz(self.joint_value_states))
        np.savez_compressed(traj_dir / "joint_vel_commands.npz", self._npz(self.joint_vel_commands))
        np.savez_compressed(traj_dir / "ee_pos_commands.npz",    self._npz(self.ee_pos_commands))

        # optional auxillaries to add later could be rewards or time stamps 

        # meta.yaml
        with open(traj_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(asdict(meta), f, sort_keys=False)
        
        self.next_idx += 1
        return traj_dir 
    
    




    


    









