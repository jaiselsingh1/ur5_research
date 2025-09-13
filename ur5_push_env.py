import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

import cc_with_cmd as cc  # Cartesian controller
from scipy.spatial.transform import Rotation 

gym.register(
    id="UR5-v1",
    entry_point="ur5_push_env:ur5",
    max_episode_steps=500,
)

class ur5(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path="./env/assets/scene.xml", frame_skip=10, **kwargs):
        super().__init__(
            model_path,
            frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32),
            **kwargs,
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[3] = np.pi
        self.init_qvel = self.data.qvel.copy()

        # joint action scaling (used as velocity limits)
        self.act_mid = np.zeros(6)
        self.act_rng = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])

        # action: [dx, dy, dz, d_rx, d_ry, d_rz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)

        # scaling 
        self.max_delta_pos = 0.1 # meters per step 
        self.max_delta_rot = 0.05    # radians per step

        # targets
        self.ee_target_pos = self.data.body("ee_finger").xpos.copy()
        self.ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        self.cartesian_controller = cc.CartesianController(self.model, self.data)
        self.ee_finger = self.data.body("ee_finger")

        # Set a fixed downward-pointing orientation
        # This creates a quaternion for pointing straight down
        # downward_rotation = Rotation.from_euler('xyz', [np.pi/2, 0, 3*np.pi/2])  
        # self.fixed_ee_quat = downward_rotation.as_quat(scalar_first=True)


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
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # translation update
        delta_pos = action[:3] * self.max_delta_pos

        # this way the target keeps accumulating 
        # self.ee_target_pos = self.ee_target_pos + delta_pos
        self.ee_target_pos = self.ee_finger.xpos.copy() + delta_pos

        # rotation update
        delta_rot = action[3:] * self.max_delta_rot
        
        # Create rotation from delta rotation vector
        delta_rotation = Rotation.from_rotvec(delta_rot)
        
        # Create rotation from current quaternion
        current_rotation = Rotation.from_quat(self.ee_finger.xquat.copy(), scalar_first=True)
        
        # Compose rotations
        new_rotation = current_rotation * delta_rotation
        
        print(self.ee_target_quat, self.ee_finger.xquat.copy())
        # Convert back to quaternion and normalize
        # self.ee_target_quat = self._normalize_quaternion(
        #      new_rotation.as_quat(scalar_first=True)
        # )

        # command to controller
        cmd = cc.Command(
            trans_x=float(self.ee_target_pos[0]),
            trans_y=float(self.ee_target_pos[1]),
            trans_z=float(self.ee_target_pos[2]),
            rot_x=float(self.ee_target_quat[1]),
            rot_y=float(self.ee_target_quat[2]),
            rot_z=float(self.ee_target_quat[3]),
            rot_w=float(self.ee_target_quat[0]),
        )

        q_current = self.data.qpos[:6].copy()

        for i in range(self.frame_skip):
            dq = self.cartesian_controller.cartesian_command(q_current, cmd)
            dq = np.clip(dq, -self.act_rng, self.act_rng)

            self.do_simulation(dq, 1)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.get_reward()

        tape_roll_xpos = self.data.body("tape_roll").xpos
        target_position = np.array([0.7, 0.2, -0.1175])
        object_to_target_error = np.linalg.norm(tape_roll_xpos - target_position)

        terminated = object_to_target_error < 0.05 # within 5 cm 
        truncated = False

        return obs, reward, terminated, truncated, {}

        
    def _get_obs(self):
        qpos = self.data.qpos[:6]
        qvel = self.data.qvel[:6]

        # in your observation you want .xpos and no .copy() since you want it at that specific state vs controllers
        ee_pos = self.data.body("ee_finger").xpos
        ee_quat = self.ee_finger.xquat
        tape_roll_pos = self.data.body("tape_roll").xpos
        target_pos = np.array([0.7, 0.2, -0.1175])

        ee_to_object = tape_roll_pos - ee_pos
        object_to_target = target_pos - tape_roll_pos
        # ee_object_distance = np.linalg.norm(ee_to_object)
        # object_target_distance = np.linalg.norm(object_to_target)
        ee_vel = self.data.body("ee_finger").cvel[:3]


        obs = np.concatenate(
            [
                qpos,
                qvel * 0.1,
                ee_pos,
                ee_quat,
                tape_roll_pos,
                target_pos,
                ee_to_object,
                object_to_target,
                # [ee_object_distance],
                # [object_target_distance],
                ee_vel * 0.1,
                self.ee_target_pos,
                self.ee_target_quat,   
            ]
        )
        return obs.astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # reset object upright

        self.set_state(qpos, qvel)
        self.ee_target_pos = self.ee_finger.xpos.copy()
        self.ee_target_quat = self.ee_finger.xquat.copy()

        return self._get_obs()

    def get_reward(self):
        tape_roll_xpos = self.data.body("tape_roll").xpos
        ee_finger_xpos = self.data.body("ee_finger").xpos
        
        ee_to_object = np.linalg.norm(ee_finger_xpos - tape_roll_xpos)
        
        reward = -10.0 * ee_to_object
        
        return reward