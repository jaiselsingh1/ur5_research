
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

import cc_with_cmd as cc  # Cartesian controller

gym.register(
    id="UR5-v1",
    entry_point="ur5_push_env:ur5",
    max_episode_steps=1000,
)

class ur5(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self, model_path="./env/assets/scene.xml", frame_skip=40, **kwargs):
        super().__init__(
            model_path,
            frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32),
            **kwargs,
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # joint action scaling (used as velocity limits)
        self.act_mid = np.zeros(6)
        self.act_rng = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)

        self.ee_target_pos = self.data.body("ee_finger").xpos.copy()
        self.ee_target_quat = self.data.body("ee_finger").xquat.copy()

        # Cartesian controller
        self.cartesian_controller = cc.CartesianController(self.model, self.data)

    def _normalize_quaternion(self, quat):
        """Ensure quaternion is normalized and handle edge cases"""
        quat = np.array(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        
        # If quaternion has zero or very small norm, return identity quaternion
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z format
        
        # Normalize the quaternion
        return quat / norm

    def step(self, action):
        action = np.clip(action, -1, 1)

        # current joint state
        q_current = self.data.qpos[:6].copy()

        # accumulate deltas
        self.ee_target_pos += np.array([
            action[0] * 0.05,
            action[1] * 0.05,
            action[2] * 0.05,
        ])

        # keep orientation fixed and ensure it's normalized
        desired_quat = self._normalize_quaternion(self.ee_target_quat)

        # command to controller
        command = cc.Command(
            trans_x=self.ee_target_pos[0],
            trans_y=self.ee_target_pos[1],
            trans_z=self.ee_target_pos[2],
            rot_x=desired_quat[1],  # x component
            rot_y=desired_quat[2],  # y component
            rot_z=desired_quat[3],  # z component
            rot_w=desired_quat[0]   # w component
        )

        try:
            dq = self.cartesian_controller.cartesian_command(q_current, command)
            dq = np.clip(dq, -self.act_rng, self.act_rng)
            self.do_simulation(dq, self.frame_skip)
        except Exception as e:
            print(f"Cartesian controller error: {e}")
            # Fall back to zero velocities if controller fails
            dq = np.zeros(6)
            self.do_simulation(dq, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.get_reward()

        tape_roll_xpos = self.data.body("tape_roll").xpos
        target_position = np.array([0.7, 0.2, -0.1175])
        object_to_target_error = np.linalg.norm(tape_roll_xpos - target_position)

        terminated = object_to_target_error < 0.05
        truncated = False

        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        qpos = self.data.qpos[:6]
        qvel = self.data.qvel[:6]

        ee_pos = self.data.body("ee_finger").xpos
        tape_roll_pos = self.data.body("tape_roll").xpos
        target_pos = np.array([0.7, 0.2, -0.1175])

        ee_to_object = tape_roll_pos - ee_pos
        object_to_target = target_pos - tape_roll_pos
        ee_object_distance = np.linalg.norm(ee_to_object)
        object_target_distance = np.linalg.norm(object_to_target)
        ee_vel = self.data.body("ee_finger").cvel[:3]

        joint_limits = np.abs(qpos) / 3.14

        obs = np.concatenate([
            qpos, qvel * 0.1,
            ee_pos, tape_roll_pos, target_pos,
            ee_to_object, object_to_target,
            [ee_object_distance], [object_target_distance],
            joint_limits, ee_vel * 0.1,
        ])
        return obs.astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[:6] += np.random.uniform(low=-0.1, high=0.1, size=6)
        qpos[9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # reset object upright

        # Reset target pose and ensure quaternion is normalized
        self.set_state(qpos, qvel)
        self.ee_target_pos = self.data.body("ee_finger").xpos.copy()
        self.ee_target_quat = self._normalize_quaternion(self.data.body("ee_finger").xquat.copy())

        return self._get_obs()

    def get_reward(self):
        try:
            tape_roll_xpos = self.data.body("tape_roll").xpos
            ee_finger_xpos = self.data.body("ee_finger").xpos

            pos_error = np.linalg.norm(ee_finger_xpos - tape_roll_xpos)
            reward = -1.0 * pos_error
            
            if pos_error < 0.05:
                reward += 500

            return reward
        
        except Exception as e:
            print(f"Reward collection error: {e}")
            return 0.0



