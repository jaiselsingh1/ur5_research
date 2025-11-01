import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class TrajMeta:
    """Metadata for a single trajectory."""
    seed: int
    workspace_bounds: Dict[str, List[float]]
    target_position: List[float]
    control_hz: float
    env_id: str
    policy_tag: str
    reward_version: str
    sim_timestep: float
    frame_skip: int
    success: bool
    final_obj_to_target: float
    steps: int
    total_reward: float = 0.0
    notes: str = ""


class TrajectoryWriter:
    """Handles writing trajectories in the required format."""
    
    def __init__(self, root_dir: str, dataset_name: str, start_idx: int = 1):
        self.root = Path(root_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.root / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.traj_idx = start_idx
        self.current_traj_dir: Optional[Path] = None
        
        # Buffers for current trajectory
        self.ee_pos_states = []
        self.ee_pos_actions = []  # Cartesian commands (x, y, z)
        self.ee_pos_commands = []  # Same as actions but explicitly named
        self.obj_pos_states = []
        self.joint_value_states = []  # qpos[:6]
        self.joint_vel_commands = []  # dq from controller
        self.rewards = []
        
    def start_traj(self):
        """Initialize a new trajectory."""
        self.current_traj_dir = self.dataset_dir / f"traj_{self.traj_idx:04d}"
        self.current_traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear buffers
        self.ee_pos_states = []
        self.ee_pos_actions = []
        self.ee_pos_commands = []
        self.obj_pos_states = []
        self.joint_value_states = []
        self.joint_vel_commands = []
        self.rewards = []
        
    def record_step(self, env, action: np.ndarray, dq_cmd: np.ndarray, reward: float):
        """Record a single step of the trajectory.
        
        Args:
            env: The MuJoCo environment
            action: Policy action (Cartesian delta in world frame, shape=(3,))
            dq_cmd: Joint velocity command from controller (shape=(6,))
            reward: Reward for this step
        """
        # End-effector position (world frame)
        ee_pos = env.ee_finger.xpos.copy()  # (3,)
        
        # Object position (world frame)
        obj_pos = env.tape_roll.xpos.copy()  # (3,)
        
        # Joint positions
        qpos = env.data.qpos[:6].copy()  # (6,)
        
        # Store data
        self.ee_pos_states.append(ee_pos)
        self.ee_pos_actions.append(action.copy())  # Should be (3,) for fix_orientation=True
        self.ee_pos_commands.append(action.copy())  # Same as action
        self.obj_pos_states.append(obj_pos)
        self.joint_value_states.append(qpos)
        self.joint_vel_commands.append(dq_cmd.copy())  # (6,)
        self.rewards.append(float(reward))
        
    def finish_traj(self, meta: TrajMeta) -> Path:
        """Save trajectory data and metadata."""
        if self.current_traj_dir is None:
            raise RuntimeError("No trajectory in progress")
        
        # Convert lists to numpy arrays
        arrays = {
            'ee_pos_states': np.array(self.ee_pos_states, dtype=np.float32),  # (T, 3)
            'ee_pos_actions': np.array(self.ee_pos_actions, dtype=np.float32),  # (T, 3)
            'ee_pos_commands': np.array(self.ee_pos_commands, dtype=np.float32),  # (T, 3)
            'obj_pos_states': np.array(self.obj_pos_states, dtype=np.float32),  # (T, 3)
            'joint_value_states': np.array(self.joint_value_states, dtype=np.float32),  # (T, 6)
            'joint_vel_commands': np.array(self.joint_vel_commands, dtype=np.float32),  # (T, 6)
            'rewards': np.array(self.rewards, dtype=np.float32),  # (T,)
        }
        
        # Calculate total reward for metadata
        meta.total_reward = float(np.sum(arrays['rewards']))
        
        # Save arrays as compressed .npz files
        for name, arr in arrays.items():
            np.savez_compressed(
                self.current_traj_dir / f"{name}.npz",
                data=arr
            )
        
        # Save metadata as YAML
        meta_dict = asdict(meta)
        with open(self.current_traj_dir / "meta.yaml", 'w') as f:
            yaml.dump(meta_dict, f, default_flow_style=False, sort_keys=False)
        
        # Increment for next trajectory
        out_dir = self.current_traj_dir
        self.traj_idx += 1
        self.current_traj_dir = None
        
        return out_dir


class TrajectoryReader:
    """Helper class to read saved trajectories."""
    
    @staticmethod
    def load_traj(traj_dir: Path) -> Dict:
        """Load a single trajectory."""
        traj_dir = Path(traj_dir)
        
        # Load metadata
        with open(traj_dir / "meta.yaml", 'r') as f:
            meta = yaml.safe_load(f)
        
        # Load arrays
        data = {}
        for npz_file in traj_dir.glob("*.npz"):
            name = npz_file.stem
            data[name] = np.load(npz_file)['data']
        
        return {'meta': meta, 'data': data}
    
    @staticmethod
    def load_dataset(dataset_dir: Path) -> List[Dict]:
        """Load all trajectories from a dataset."""
        dataset_dir = Path(dataset_dir)
        trajectories = []
        
        for traj_dir in sorted(dataset_dir.glob("traj_*")):
            if traj_dir.is_dir():
                trajectories.append(TrajectoryReader.load_traj(traj_dir))
        
        return trajectories