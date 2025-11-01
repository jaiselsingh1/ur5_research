"""
Data collection script for UR5 push environment.
Generates datasets in the format specified by Lennart Schulze.

Each trajectory includes:
- ee_pos_states.npz: End-effector positions (T, 3)
- ee_pos_actions.npz: Policy actions = Cartesian commands (T, 3)
- ee_pos_commands.npz: Same as actions (T, 3)
- obj_pos_states.npz: Object positions (T, 3)
- joint_value_states.npz: Joint positions (T, 6)
- joint_vel_commands.npz: Joint velocity commands from controller (T, 6)
- rewards.npz: Per-step rewards (T,)
- meta.yaml: Trajectory metadata
"""

from pathlib import Path
import imageio
import numpy as np
from sbx import PPO
import ur5_push_env
from traj_logger import TrajectoryWriter, TrajMeta


def collect_dataset(
    policy_path: str = "./logs/best_model.zip",
    out_root: str = "./saved_data",
    dataset_name: str = "ur5_push_100traj",
    num_rollouts: int = 100,
    steps_per_rollout: int = 500,
    save_video: bool = False,
    video_fps: int = 10,
    deterministic: bool = True,
    seed0: int = 42,
):
    """Collect trajectories from a trained policy.
    
    Args:
        policy_path: Path to trained policy checkpoint
        out_root: Root directory for saved data
        dataset_name: Name of this dataset (creates a subfolder)
        num_rollouts: Number of trajectories to collect
        steps_per_rollout: Max steps per trajectory
        save_video: Whether to save videos (not required per conversation)
        video_fps: Video frame rate (should match control_hz = 10)
        deterministic: Use deterministic policy actions
        seed0: Starting random seed
    """
    # Create environment - IMPORTANT: frame_skip should give 10Hz control
    env = ur5_push_env.ur5(
        render_mode="rgb_array" if save_video else None,
        frame_skip=50,  # With timestep=0.002, this gives 10Hz
        fix_orientation=True
    )
    
    # Verify control frequency
    control_hz = 1.0 / (env.model.opt.timestep * env.frame_skip)
    assert abs(control_hz - 10.0) < 0.5, \
        f"Control Hz is {control_hz:.1f}, expected 10 Hz (timestep={env.model.opt.timestep}, frame_skip={env.frame_skip})"
    
    policy = PPO.load(policy_path)
    traj_writer = TrajectoryWriter(out_root, dataset_name, start_idx=1)
    
    if save_video:
        video_dir = traj_writer.dataset_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Workspace bounds (from your cmd clipping)
    workspace_bounds = {
        "x": [0.502 - 0.3, 0.502 + 0.3],
        "y": [-0.6, 0.6],
        "z": [0.03, 0.30],
    }
    
    # Collection loop
    for k in range(num_rollouts):
        # Reset environment
        obs, info = env.reset(seed=seed0 + k)
        traj_writer.start_traj()
        
        # Open video writer if needed
        video_writer = None
        if save_video:
            video_path = video_dir / f"rollout_{k:04d}.mp4"
            video_writer = imageio.get_writer(
                str(video_path), 
                fps=video_fps,
                codec="libx264",
                quality=8
            )
        
        # Trajectory bookkeeping
        success_flag = False
        final_err = None
        total_reward = 0.0
        
        # Rollout loop
        for t in range(steps_per_rollout):
            # Get action from policy
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, rdict = env.step(action)
            total_reward += reward
            
            # Get joint velocity command (what was actually sent to robot)
            dq_cmd = getattr(env, "last_dq_cmd", np.zeros(6, dtype=np.float32))
            
            # Record step (at 10Hz as per environment control frequency)
            traj_writer.record_step(
                env=env,
                action=action,  # (3,) Cartesian commands in world frame
                dq_cmd=dq_cmd,  # (6,) joint velocities
                reward=reward
            )
            
            # Save video frame if needed
            if video_writer is not None:
                frame = env.render()
                video_writer.append_data(frame)
            
            # Check termination
            if terminated or truncated:
                obj_pos = env.tape_roll.xpos
                target_pos = env.target_position
                final_err = float(np.linalg.norm(obj_pos - target_pos))
                success_flag = bool(terminated)
                break
        
        # Close video
        if video_writer is not None:
            video_writer.close()
        
        # Create metadata
        meta = TrajMeta(
            seed=seed0 + k,
            workspace_bounds=workspace_bounds,
            target_position=list(env.target_position.astype(float)),
            control_hz=float(control_hz),
            env_id="UR5-v1",
            policy_tag=Path(policy_path).stem,
            reward_version="v0.1",
            sim_timestep=float(env.model.opt.timestep),
            frame_skip=int(env.frame_skip),
            success=success_flag,
            final_obj_to_target=(final_err if final_err is not None else -1.0),
            steps=t + 1,
            total_reward=total_reward,
            notes="Commands are Cartesian EE positions (x,y,z) in world frame; " \
                  "actions are deltas, commands are target positions"
        )
        
        # Save trajectory
        out_dir = traj_writer.finish_traj(meta)
    
    env.close()


def verify_dataset(dataset_dir: str) -> dict:
    """Verify the collected dataset has the expected format.
    
    Returns:
        dict with 'valid': bool and 'info': dict with stats
    """
    from traj_logger import TrajectoryReader
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return {'valid': False, 'error': f"Dataset directory not found: {dataset_dir}"}
    
    trajectories = TrajectoryReader.load_dataset(dataset_path)
    
    if len(trajectories) == 0:
        return {'valid': False, 'error': "No trajectories found"}
    
    # Check first trajectory
    traj = trajectories[0]
    meta = traj['meta']
    data = traj['data']
    
    # Verify required files
    required = [
        'ee_pos_states', 'ee_pos_actions', 'ee_pos_commands',
        'obj_pos_states', 'joint_value_states', 'joint_vel_commands',
        'rewards'
    ]
    missing = [r for r in required if r not in data]
    
    return {
        'valid': len(missing) == 0,
        'num_trajectories': len(trajectories),
        'missing_arrays': missing,
        'sample_meta': meta,
        'sample_shapes': {k: v.shape for k, v in data.items()}
    }


if __name__ == "__main__":
    # Collect 100 trajectories
    collect_dataset(
        policy_path="./logs/best_model.zip",
        out_root="./saved_data",
        dataset_name="ur5_push_100traj",
        num_rollouts=100,
        steps_per_rollout=500,
        save_video=False,
        deterministic=True,
        seed0=42,
    )
    
    # can verify
    # result = verify_dataset("./saved_data/ur5_push_100traj")
    # assert result['valid'], f"Verification failed: {result}"
    
    # Collect 10k trajectories dataset
    # collect_dataset(
    #     dataset_name="ur5_push_10k",
    #     num_rollouts=10000,
    #     save_video=False,
    #     seed0=1000,
    # )