"""
Data collection script for UR5 push environment.

Each trajectory includes:
- ee_pos_states.npz: End-effector positions (T, 3)
- ee_pos_actions.npz: Deltas = commands - states (T, 3)
- ee_pos_commands.npz: Cartesian commands from policy (T, 3)
- obj_pos_states.npz: Object positions (T, 3)
- joint_value_states.npz: Joint positions (T, 6)
- joint_vel_commands.npz: Joint velocity commands from controller (T, 6)
- rewards.npz: Per-step rewards (T,)
- meta.yaml: Trajectory metadata
- cam0/, cam1/: Images from two camera views (optional)
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
        policy_path: Path to trained PPO model
        out_root: Root directory for datasets
        dataset_name: Name of this dataset
        num_rollouts: Number of trajectories to collect
        steps_per_rollout: Max steps per trajectory
        save_video: If True, save images to cam0/ and cam1/ folders
        video_fps: FPS for combined video (should be 10 Hz)
        deterministic: Use deterministic policy actions
        seed0: Starting seed (increments for each rollout)
    """
    
    env = ur5_push_env.ur5(
        render_mode="rgb_array" if save_video else None,
        frame_skip=50,
        fix_orientation=True,
        width=1280,  # Use 1280x720 resolution
        height=720
    )
    
    control_hz = 1.0 / (env.model.opt.timestep * env.frame_skip)
    
    policy = PPO.load(policy_path)
    
    # Workspace bounds from environment
    workspace_bounds = {
        "x": [0.502 - 0.3, 0.502 + 0.3],
        "y": [-0.6, 0.6],
        "z": [0.03, 0.30],
    }
    
    traj_writer = TrajectoryWriter(
        root_dir=out_root,
        dataset_name=dataset_name,
        start_idx=1
    )
        
    for k in range(num_rollouts):
        seed = seed0 + k
        obs, info = env.reset(seed=seed)
        
        traj_writer.start_traj()
        
        # hide target position by setting geom alpha = 0 
        if save_video:
            # Disable target visualization if it exists
            target_geom_id = env.model.geom("target_position").id if "target_position" in [env.model.geom(i).name for i in range(env.model.ngeom)] else None
            if target_geom_id is not None:
                env.model.geom_rgba[target_geom_id, 3] = 0.0  # Set alpha to 0
        
        # Setup video writers for two camera views
        cam0_vw = None
        cam1_vw = None
        cam0_dir = None
        cam1_dir = None
        
        if save_video:
            cam0_dir = traj_writer.current_traj_dir / "cam0"
            cam1_dir = traj_writer.current_traj_dir / "cam1"
            cam0_dir.mkdir(exist_ok=True)
            cam1_dir.mkdir(exist_ok=True)
            
            cam0_vw = imageio.get_writer(
                cam0_dir / "__combined.mp4",
                fps=video_fps,
                codec="libx264",
                quality=8
            )
            cam1_vw = imageio.get_writer(
                cam1_dir / "__combined.mp4",
                fps=video_fps,
                codec="libx264",
                quality=8
            )
        
        success_flag = False
        final_err = None
        
        for t in range(steps_per_rollout):
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            # Store EE position before step (for computing deltas)
            ee_pos_before = env.ee_finger.xpos.copy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store EE position after step
            ee_pos_after = env.ee_finger.xpos.copy()
            
            # Get joint velocity command from controller
            dq_cmd = env.last_dq_cmd if hasattr(env, 'last_dq_cmd') else np.zeros(6)
            
            # ee_pos_actions: commands - states
            # The policy outputs a command (target position or delta)
            # We need to extract what the command actually was
            # If fix_orientation=True, action is (3,) representing desired EE position change
            ee_pos_command = action[:3] if len(action) >= 3 else action
            
            # Record step with proper action calculation
            traj_writer.record_step(
                env=env,
                action=ee_pos_command,  # Policy command
                ee_pos_before=ee_pos_before,  # For computing actual delta
                ee_pos_after=ee_pos_after,
                dq_cmd=dq_cmd,
                reward=reward
            )
            
            # Save images from both camera views at 10Hz 
            if save_video:
                # side camera
                env.mujoco_renderer.camera_name = "side_cam"
                frame_cam0 = env.render()
                imageio.imwrite(
                    cam0_dir / f"test_render_{t:06d}_side.png",
                    frame_cam0
                )
                cam0_vw.append_data(frame_cam0)
                
                # birdseye tilted camera 
                env.mujoco_renderer.camera_name = "birdseye_tilted_cam"
                frame_cam1 = env.render()
                imageio.imwrite(
                    cam1_dir / f"test_render_{t:06d}_birdseye.png",
                    frame_cam1
                )
                cam1_vw.append_data(frame_cam1)
            
            if terminated or truncated:
                success_flag = bool(terminated)
                final_err = float(info.get('final_distance', -1.0))
                break
        
        # Close video writers
        if save_video:
            cam0_vw.close()
            cam1_vw.close()
        
        meta = TrajMeta(
            seed=seed,
            workspace_bounds=workspace_bounds,
            target_position=[float(x) for x in env.target_position],  # Convert to native list
            control_hz=float(control_hz),
            env_id="UR5-v1",
            policy_tag=Path(policy_path).name,
            reward_version="v0.1",
            sim_timestep=float(env.model.opt.timestep),
            frame_skip=int(env.frame_skip),
            success=success_flag,
            final_obj_to_target=float(final_err) if final_err is not None else -1.0,
            steps=t + 1 if 't' in locals() else 0,
            notes="",
        )
        
        out_dir = traj_writer.finish_traj(meta)
    
    env.close()

if __name__ == "__main__":
    collect_dataset(
        policy_path="./logs/best_model.zip",
        out_root="./saved_data",
        dataset_name="ur5_push_test",
        num_rollouts=10,
        save_video=True,
        deterministic=True
    )