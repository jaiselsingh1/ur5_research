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

    env = ur5_push_env.ur5(
        render_mode="rgb_array" if save_video else None,
        frame_skip=50,
        fix_orientation=True
    )
    
    control_hz = 1.0 / (env.model.opt.timestep * env.frame_skip)
    
    policy = PPO.load(policy_path)
    traj_writer = TrajectoryWriter(out_root, dataset_name, start_idx=1)
    
    workspace_bounds = {
        "x": [0.502 - 0.3, 0.502 + 0.3],
        "y": [-0.6, 0.6],
        "z": [0.03, 0.30],
    }
    
    for k in range(num_rollouts):
        obs, info = env.reset(seed=seed0 + k)
        traj_writer.start_traj()
        
        cam0_dir = None
        cam1_dir = None
        video_writer_cam0 = None
        video_writer_cam1 = None
        
        if save_video:
            cam0_dir = traj_writer.current_traj_dir / "cam0"
            cam0_dir.mkdir(exist_ok=True)
            combined_video_path_cam0 = cam0_dir / "__combined.mp4"
            video_writer_cam0 = imageio.get_writer(
                str(combined_video_path_cam0), 
                fps=video_fps,
                codec="libx264",
                quality=8
            )
            
            cam1_dir = traj_writer.current_traj_dir / "cam1"
            cam1_dir.mkdir(exist_ok=True)
            combined_video_path_cam1 = cam1_dir / "__combined.mp4"
            video_writer_cam1 = imageio.get_writer(
                str(combined_video_path_cam1), 
                fps=video_fps,
                codec="libx264",
                quality=8
            )
        
        success_flag = False
        final_err = None
        total_reward = 0.0
        
        for t in range(steps_per_rollout):
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, rdict = env.step(action)
            total_reward += reward
            
            dq_cmd = getattr(env, "last_dq_cmd", np.zeros(6, dtype=np.float32))
            
            traj_writer.record_step(
                env=env,
                action=action,
                dq_cmd=dq_cmd,
                reward=reward
            )
            
            if cam0_dir is not None:
                env.mujoco_renderer.camera_name = "birdseye_tilted_cam"
                frame_cam0 = env.render()
                image_path_cam0 = cam0_dir / f"test_render_{t:06d}_top.png"
                imageio.imwrite(image_path_cam0, frame_cam0)
                if video_writer_cam0 is not None:
                    video_writer_cam0.append_data(frame_cam0)
            
            if cam1_dir is not None:
                env.mujoco_renderer.camera_name = "side_cam"
                frame_cam1 = env.render()
                image_path_cam1 = cam1_dir / f"test_render_{t:06d}_side.png"
                imageio.imwrite(image_path_cam1, frame_cam1)
                if video_writer_cam1 is not None:
                    video_writer_cam1.append_data(frame_cam1)
            
            if terminated or truncated:
                obj_pos = env.tape_roll.xpos
                target_pos = env.target_position
                final_err = float(np.linalg.norm(obj_pos - target_pos))
                success_flag = bool(terminated)
                break
        
        if video_writer_cam0 is not None:
            video_writer_cam0.close()
        if video_writer_cam1 is not None:
            video_writer_cam1.close()
        
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
        
        out_dir = traj_writer.finish_traj(meta)
    
    env.close()


if __name__ == "__main__":
    collect_dataset(
        policy_path="./logs/best_model.zip",
        out_root="./saved_data",
        dataset_name="ur5_push_100traj",
        num_rollouts=100,
        steps_per_rollout=500,
        save_video=True,
        deterministic=True,
        seed0=42,
    )