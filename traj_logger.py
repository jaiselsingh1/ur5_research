"""
Trajectory logging utilities for UR5 dataset collection.

Handles writing trajectories in the format required by Lennart Schulze:
- Multiple .npz files containing trajectory data
- meta.yaml with trajectory metadata
- Optional image sequences from multiple camera views
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class TrajMeta:
    """Metadata for a single trajectory.
    
    Fields:
    - seed: Random seed used for this trajectory's environment initialization
    - workspace_bounds: Dictionary defining the valid (x, y, z) workspace limits for the end-effector
    - target_position: The (x, y, z) goal position that the object should be pushed to
    - control_hz: Control frequency in Hz (should be 10.0 for this dataset)
    - env_id: Gymnasium environment ID string
    - policy_tag: Filename of the policy model used to generate this trajectory
    - reward_version: Version string for the reward function used
    - sim_timestep: MuJoCo simulation timestep in seconds
    - frame_skip: Number of simulation steps per control step
    - success: Boolean indicating whether the trajectory successfully reached the goal
    - final_obj_to_target: Final distance from object to target position at trajectory end
    - steps: Total number of control steps in this trajectory
    - total_reward: Sum of rewards over the entire trajectory
    - notes: Optional text field for additional information
    """
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
        """Initialize trajectory writer.
        
        Args:
            root_dir: Root directory for all datasets
            dataset_name: Name of this specific dataset
            start_idx: Starting index for trajectory numbering
        """
        self.root = Path(root_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.root / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.traj_idx = start_idx
        self.current_traj_dir: Optional[Path] = None
        
        # Buffers for current trajectory
        self.ee_pos_states = []  # Actual EE positions (after step)
        self.ee_pos_commands = []  # Absolute commanded positions from policy
        self.obj_pos_states = []
        self.joint_value_states = []
        self.joint_vel_commands = []
        self.rewards = []
        
    def start_traj(self):
        """Initialize a new trajectory."""
        self.current_traj_dir = self.dataset_dir / f"traj_{self.traj_idx:04d}"
        self.current_traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear buffers
        self.ee_pos_states = []
        self.ee_pos_commands = []
        self.obj_pos_states = []
        self.joint_value_states = []
        self.joint_vel_commands = []
        self.rewards = []
        
    def record_step(
        self,
        env,
        ee_pos_command: np.ndarray,
        ee_pos_after: np.ndarray,
        dq_cmd: np.ndarray,
        reward: float
    ):
        """Record a single step of the trajectory.
        
        Args:
            env: The MuJoCo environment
            ee_pos_command: Absolute commanded EE position (3,)
            ee_pos_after: Actual EE position after this step (3,)
            dq_cmd: Joint velocity command from controller (shape=(6,))
            reward: Reward for this step
        """
        # Store EE state (actual position after step)
        self.ee_pos_states.append(ee_pos_after.copy())
        
        # Store absolute commanded position
        self.ee_pos_commands.append(ee_pos_command.copy())
        
        # Object position
        obj_pos = env.tape_roll.xpos.copy()
        self.obj_pos_states.append(obj_pos)
        
        # Joint positions
        qpos = env.data.qpos[:6].copy()
        self.joint_value_states.append(qpos)
        
        # Joint velocity commands
        self.joint_vel_commands.append(dq_cmd.copy())
        
        # Reward
        self.rewards.append(float(reward))
        
    def finish_traj(self, meta: TrajMeta) -> Path:
        """Save trajectory data and metadata.
        
        Returns:
            Path to the saved trajectory directory
        """
        if self.current_traj_dir is None:
            raise RuntimeError("No trajectory in progress")
        
        # Convert lists to numpy arrays
        ee_pos_states = np.array(self.ee_pos_states, dtype=np.float64)  # (T, 3)
        ee_pos_commands = np.array(self.ee_pos_commands, dtype=np.float64)  # (T, 3)
        obj_pos_states = np.array(self.obj_pos_states, dtype=np.float64)  # (T, 3)
        joint_value_states = np.array(self.joint_value_states, dtype=np.float64)  # (T, 6)
        joint_vel_commands = np.array(self.joint_vel_commands, dtype=np.float64)  # (T, 6)
        rewards = np.array(self.rewards, dtype=np.float64)  # (T,)
        
        # ee_pos_actions = tracking error between commanded and actual positions
        # ee_pos_commands = absolute desired positions from policy
        # ee_pos_states = actual observed positions after step
        ee_pos_actions = ee_pos_commands - ee_pos_states  # (T, 3)
        
        # Calculate total reward for metadata
        meta.total_reward = float(np.sum(rewards))
        
        # Save arrays as compressed .npz files
        # Each file contains a single array with a descriptive key name
        np.savez_compressed(
            self.current_traj_dir / "ee_pos_states.npz",
            ee_pos_states=ee_pos_states
        )
        np.savez_compressed(
            self.current_traj_dir / "ee_pos_actions.npz",
            ee_pos_actions=ee_pos_actions  # Tracking error: commands - states
        )
        np.savez_compressed(
            self.current_traj_dir / "ee_pos_commands.npz",
            ee_pos_commands=ee_pos_commands
        )
        np.savez_compressed(
            self.current_traj_dir / "obj_pos_states.npz",
            obj_pos_states=obj_pos_states
        )
        np.savez_compressed(
            self.current_traj_dir / "joint_value_states.npz",
            joint_value_states=joint_value_states
        )
        np.savez_compressed(
            self.current_traj_dir / "joint_vel_commands.npz",
            joint_vel_commands=joint_vel_commands
        )
        np.savez_compressed(
            self.current_traj_dir / "rewards.npz",
            rewards=rewards
        )
        
        meta_dict = asdict(meta)
        
        # Ensure all numpy types are converted to native Python types
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        meta_dict = convert_to_native(meta_dict)
        
        # Save metadata as YAML
        with open(self.current_traj_dir / "meta.yaml", 'w') as f:
            yaml.dump(meta_dict, f, default_flow_style=False, sort_keys=False)
        
        # Increment for next trajectory
        out_dir = self.current_traj_dir
        self.traj_idx += 1
        self.current_traj_dir = None
        
        return out_dir