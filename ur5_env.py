from os import path
import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from email.quoprimime import body_check

"""
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}"""

gym.register(
    id ="UR5-v0",
    entry_point ="ur5_env:ur5",
    max_episode_steps=5000,
)

# custom MuJoCo environment in Gymnasium
class ur5(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        model_path="./env/assets/scene.xml",
        frame_skip = 40,
        # robot_noise_ratio: float = 0.01,
        # default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        # path to the relative directory mapping
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path
        )

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        # the super here is the MuJoCo env vs the Gymnasium Env
        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            # default_camera_config=default_camera_config,
            **kwargs,
            # *args or **kwargs -> * is used in python in order to expand tuples or lists or dictionaries
        )

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        num_actuators = 6 

        # self.act_mid = np.zeros(num_actuators)
        # self.act_rng = np.ones(num_actuators) * 0.5 # reduced to scale down action scaling 
        self.act_mid = np.zeros(num_actuators)
        self.act_rng = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])  # XML ctrlrange

        # this makes more sense when you scale it here vs the neural network because then you don't have to write that scalar multiplier for each output
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float64)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # de-normalize the input action to the needed control range
        # this is basically taking the range from [-1,1] and then scaling it for the joints to be able to move around and meet a reward
        action = self.act_mid + action * self.act_rng

        # enforce vel limits
        # ctrl_feasible = self._ctrl_velocity_limits(action)
        # enforce position limits
        # ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.get_reward()

        # termination conditions based on the task
        tape_roll_xpos = self.data.body("tape_roll").xpos
        ee_finger_xpos = self.data.body("ee_finger").xpos

        pos_error = np.linalg.norm(ee_finger_xpos - tape_roll_xpos)
        terminated = pos_error < 0.05  # Success if within 5cm

        truncated = False
        

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos = self.data.qpos[:6]  # Only UR5 joints
        qvel = self.data.qvel[:6]  # Only UR5 velocities
        
        ee_pos = self.data.body("ee_finger").xpos
        target_pos = self.data.body("tape_roll").xpos
        # Calculate relative position and distance
        relative_pos = target_pos - ee_pos
        distance = np.linalg.norm(relative_pos)

        # Get end-effector velocity
        ee_vel = self.data.body("ee_finger").cvel[:3]
        
        # Normalized joint limits (how close to limits)
        joint_limits = np.abs(qpos) / 3.14  # Normalized by ±pi
        
        # Concatenate all observations
        obs = np.concatenate([
            qpos,                    # 6
            qvel * 0.1,             # 6 (scaled for stability)
            ee_pos,                  # 3
            target_pos,              # 3
            relative_pos,            # 3
            [distance],              # 1
            joint_limits,            # 6
            ee_vel * 0.1,           # 3 (scaled)
        ])
        
        return obs.astype(np.float32)


    def reset_model(self):
        # Reset UR5 to initial position with some randomization
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # Add small random noise to UR5 joint positions (first 6)
        qpos[:6] += np.random.uniform(low=-0.1, high=0.1, size=6)
        
        # Randomize tape roll position slightly
        tape_roll_base_pos = np.array([0.502, -0.05, -0.1175])  # from xml
        # qpos[6:9] = tape_roll_base_pos + np.random.uniform(low=-0.05, high=0.05, size=3)
        
        # Reset tape roll orientation (quaternion) - keep it upright
        qpos[9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z quaternion
        
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs

    def get_reward(self):

        #tape_roll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tape_roll")
        #end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_finger")

        # enum is a data structure that has different types, that has named access
        try:
            tape_roll_xpos = self.data.body("tape_roll").xpos
            ee_finger_xpos = self.data.body("ee_finger").xpos

            pos_error = np.linalg.norm(ee_finger_xpos - tape_roll_xpos)
            reward = -1.0 * pos_error

        
            # if self.data.time % 1.0 < 0.04:  
            #     print(f"EE pos: {ee_finger_xpos}")
            #     print(f"Target pos: {tape_roll_xpos}")
            #     print(f"Distance: {pos_error:.3f}, Reward: {reward:.2f}")
            return reward
        
        except Exception as e:
            print(f"Reward collection error: {e}")
            return 0.0 



"""
    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray):
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:9, 0], self.robot_pos_bound[:9, 1]
        )
        return ctrl_feasible_vel

    def _ctrl_position_limits(self, ctrl_position: np.ndarray):
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:9, 0], self.robot_pos_bound[:9, 1]
        )
        return ctrl_fesible_position
"""
