import time
from pathlib import Path

import imageio
import numpy as np
from sbx import PPO

import ur5_push_env
from traj_logger import TrajectoryWriter, TrajMeta 

import time
from pathlib import Path

import imageio
import numpy as np
from sbx import PPO

import ur5_push_env
from traj_logger import TrajectoryWriter, TrajMeta  # your class file


def collect(
    policy_path: str = "./logs/best_model.zip",
    out_root: str = "./saved_data",
    dataset_name: str = "ur5_push",
    num_rollouts: int = 10,
    steps_per_rollout: int = 500,
    save_video: bool = True,
    video_fps: int = 30,
    deterministic: bool = True,
    seed0: int = 0,
):
    # env and policy 
    env = ur5_push_env.ur5(render_mode="rgb_array", fix_orientation=True)
    policy = PPO.load(policy_path)

    # write config
    traj_writer = TrajectoryWriter(out_root, dataset_name, start_idx=1)
    if save_video:
        video_dir = Path(out_root) / dataset_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    # convenience: bounds you already enforce in the cmd clip
    workspace_bounds = {
        "x": [0.502 - 0.3, 0.502 + 0.3],
        "y": [-0.6, 0.6],
        "z": [0.03, 0.30],  # cmd clamping
    }

    for k in range(num_rollouts):
        # reset and book keeping 
        obs, info = env.reset(seed=seed0 + k)
        traj_writer.start_traj()

        # optionally open a video for this rollout
        if save_video:
            vw = imageio.get_writer(
                str(video_dir / f"rollout_{k:04d}.mp4"), fps=video_fps, codec="libx264"
            )

        success_flag = False
        final_err = None

        for t in range(steps_per_rollout):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, rdict = env.step(action)

            dq_cmd = getattr(env, "last_dq_cmd", np.zeros(6, dtype=np.float32))
            traj_writer.record_step(env=env, action=action, dq_cmd=dq_cmd, reward=reward)

            if save_video:
                frame = env.render()
                vw.append_data(frame)

            if terminated or truncated:
                # track the final object-to-target error if present
                obj = env.tape_roll.xpos
                target = env.target_position
                final_err = float(np.linalg.norm(obj - target))
                success_flag = bool(terminated)  # terminated encodes sucess 
                break

       
        if save_video:
            vw.close()

        # meta and npz bundle 
        # fill meta from env and run
        meta = TrajMeta(
            seed=seed0 + k,
            workspace_bounds=workspace_bounds,
            target_position=list(env.target_position.astype(float)),
            control_hz=float(1.0 / (env.model.opt.timestep * env.frame_skip)),
            env_id="UR5-v1",
            policy_tag=Path(policy_path).name,
            reward_version="v0.1",  # can be bumped 
            sim_timestep=float(env.model.opt.timestep),
            frame_skip=float(env.frame_skip),
            success=success_flag,
            final_obj_to_target=(final_err if final_err is not None else -1.0),
            steps=t + 1 if ('t' in locals()) else 0,
            notes="",
        )
        out_dir = traj_writer.finish_traj(meta)
        # print(f"[OK] saved rollout {k+1}/{num_rollouts} to {out_dir}")

    env.close()


if __name__ == "__main__":
    # add tyro in the future 
    collect()

