import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import cc_with_cmd as cc  # Cartesian controller


class MJX_UR5(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self, model_path="./env/assets/scene.xml", frame_skip=40):
        super().__init__()

        # Load mujoco model + mjx system
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_sys = mjx.put_model(self.mj_model)

        # initial mjx data
        data0 = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, data0)
        self.mjx_state = mjx.put_data(self.mj_model, data0)

        self.frame_skip = frame_skip

        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(41,), dtype=jnp.float32
        )

        # Action space: Cartesian deltas
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=jnp.float64)

        # scaling
        self.max_delta_pos = 0.05
        self.max_delta_rot = 0.1

        # initial targets
        ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ee_finger")
        self.ee_body_id = ee_body_id

        self.ee_target_pos = self.mjx_state.xpos[ee_body_id]
        self.ee_target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

        # Cartesian controller
        self.cartesian_controller = cc.CartesianController(self.mj_model, data0)

    def _normalize_quat(self, quat):
        norm = jnp.linalg.norm(quat)
        return jnp.where(norm < 1e-8, jnp.array([1.0, 0.0, 0.0, 0.0]), quat / norm)

    def step(self, action):
        action = jnp.clip(action, -1.0, 1.0)

        # update targets
        delta_pos = action[:3] * self.max_delta_pos
        self.ee_target_pos = self.ee_target_pos + delta_pos

        delta_rot = action[3:] * self.max_delta_rot
        delta_rotation = R.from_rotvec(delta_rot)
        current_rotation = R.from_quat(
            jax.device_get(self.mjx_state.xquat[self.ee_body_id])
        )
        new_rotation = current_rotation * delta_rotation
        self.ee_target_quat = self._normalize_quat(new_rotation.as_quat())

        # build controller command
        cmd = cc.Command(
            trans_x=float(self.ee_target_pos[0]),
            trans_y=float(self.ee_target_pos[1]),
            trans_z=float(self.ee_target_pos[2]),
            rot_x=float(self.ee_target_quat[1]),
            rot_y=float(self.ee_target_quat[2]),
            rot_z=float(self.ee_target_quat[3]),
            rot_w=float(self.ee_target_quat[0]),
        )

        # roll out dynamics with frame_skip
        q_current = jax.device_get(self.mjx_state.qpos[:6])
        for _ in range(self.frame_skip):
            dq = self.cartesian_controller.cartesian_command(q_current, cmd)
            dq = jnp.clip(dq, -3.15, 3.15)
            self.mjx_state = mjx.step(self.mjx_sys, self.mjx_state, dq)

        obs = self._get_obs()
        reward = self.get_reward()

        tape_roll_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tape_roll"
        )
        tape_roll_xpos = self.mjx_state.xpos[tape_roll_id]
        target_position = jnp.array([0.7, 0.2, -0.1175])

        terminated = jnp.linalg.norm(tape_roll_xpos - target_position) < 0.05
        truncated = False

        return obs, float(reward), bool(terminated), bool(truncated), {}

    def _get_obs(self):
        qpos = self.mjx_state.qpos[:6]
        qvel = self.mjx_state.qvel[:6]
        ee_pos = self.mjx_state.xpos[self.ee_body_id]
        ee_quat = self.mjx_state.xquat[self.ee_body_id]

        tape_roll_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tape_roll"
        )
        tape_roll_pos = self.mjx_state.xpos[tape_roll_id]
        target_pos = jnp.array([0.7, 0.2, -0.1175])

        ee_to_object = tape_roll_pos - ee_pos
        object_to_target = target_pos - tape_roll_pos
        ee_vel = self.mjx_state.cvel[self.ee_body_id, :3]

        obs = jnp.concatenate(
            [
                qpos,
                qvel * 0.1,
                ee_pos,
                ee_quat,
                tape_roll_pos,
                target_pos,
                ee_to_object,
                object_to_target,
                ee_vel * 0.1,
                self.ee_target_pos,
                self.ee_target_quat,
            ]
        )
        return jnp.array(obs, dtype=jnp.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        data0 = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, data0)
        self.mjx_state = mjx.put_data(self.mj_model, data0)

        self.ee_target_pos = self.mjx_state.xpos[self.ee_body_id]
        self.ee_target_quat = self.mjx_state.xquat[self.ee_body_id]

        return self._get_obs(), {}

    def get_reward(self):
        tape_roll_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tape_roll"
        )
        tape_roll_xpos = self.mjx_state.xpos[tape_roll_id]
        ee_pos = self.mjx_state.xpos[self.ee_body_id]

        ee_to_object = jnp.linalg.norm(ee_pos - tape_roll_xpos)
        reward = -10.0 * ee_to_object
        return reward




        
    





