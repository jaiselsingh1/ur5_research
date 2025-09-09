from os import path
import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from email.quoprimime import body_check
from scipy.spatial.transform import Rotation as R
import IK

"""
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}"""

gym.register(
    id ="UR5-v1",
    entry_point ="ur5_push_env:ur5",
    max_episode_steps=1000,
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

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)

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

        self.IK = IK.GradientDescentIK(self.model, self.data, 0.5, 0.01, 0.5, )

    def _cartesian_to_joint_velocity(self, action):
        # Scale the action to desired delta pose
        delta_pos = action[:3] * 0.05  
        delta_rot = action[3:] * 0.2  

        ee_id = self.model.body("ee_finger").id
        current_pos = self.data.body(ee_id).xpos.copy()
        current_quat = self.data.body(ee_id).xquat.copy()

        # compute goal position
        goal_pos = current_pos + delta_pos

        # Convert delta_rot (axis-angle) to quaternion
        rotvec = delta_rot  # already in radians
        delta_quat = R.from_rotvec(rotvec).as_quat()  # [x,y,z,w]
        delta_quat = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])  # reorder to [w,x,y,z]

        # Compose goal quaternion
        goal_quat = self._quat_multiply(current_quat, delta_quat)

        # Current joint state
        q_current = self.data.qpos[:6].copy()

        # Solve IK
        q_next = self.IK.calculate(goal_pos, q_current, ee_id, goal_quat)

        if q_next is None:
            # IK failed: return zero velocities (no movement)
            return np.zeros(6)

        # Joint velocity command (delta q / dt)
        dt = self.model.opt.timestep * self.frame_skip
        joint_velocities = (q_next - q_current) / dt

        # Clip within actuator limits from the XML 
        vel_limits = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])
        return np.clip(joint_velocities, -vel_limits, vel_limits)
    

    def _quat_multiply(self, q1, q2):
        """Quaternion multiplication, [w,x,y,z] convention."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def step(self, action):

        action = np.clip(action, -1, 1)
        # de-normalize the input action to the needed control range
        # this is basically taking the range from [-1,1] and then scaling it for the joints to be able to move around and meet a reward
        # action = self.act_mid + action * self.act_rng

        action = self._cartesian_to_joint_velocity(action)

        # enforce vel limits
        # ctrl_feasible = self._ctrl_velocity_limits(action)
        # enforce position limits
        # ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.get_reward()

        # termination conditions 
        tape_roll_xpos = self.data.body("tape_roll").xpos
        target_position = np.array([0.7, 0.2, -0.1175])
        object_to_target_error = np.linalg.norm(tape_roll_xpos - target_position)
    
        terminated = object_to_target_error < 0.05  # Success when object within 5cm of target
        truncated = False
    
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos = self.data.qpos[:6]  # Only UR5 joints
        qvel = self.data.qvel[:6]  # Only UR5 velocities
        
        ee_pos = self.data.body("ee_finger").xpos
        tape_roll_pos = self.data.body("tape_roll").xpos
        target_pos = np.array([0.7, 0.2, -0.1175])  # Target position for tape roll
        
        # Calculate various relative positions
        ee_to_object = tape_roll_pos - ee_pos
        object_to_target = target_pos - tape_roll_pos
        ee_to_target = target_pos - ee_pos
        
        # Distances
        ee_object_distance = np.linalg.norm(ee_to_object)
        object_target_distance = np.linalg.norm(object_to_target)
        
        # Get end-effector velocity
        ee_vel = self.data.body("ee_finger").cvel[:3]
        
        # Normalized joint limits
        joint_limits = np.abs(qpos) / 3.14
        
        # Concatenate all observations
        obs = np.concatenate([
            qpos,                      # 6
            qvel * 0.1,               # 6 (scaled for stability)
            ee_pos,                   # 3
            tape_roll_pos,            # 3
            target_pos,               # 3 - NEW: target position
            ee_to_object,             # 3 - relative position ee to object
            object_to_target,         # 3 - NEW: relative position object to target
            [ee_object_distance],     # 1
            [object_target_distance], # 1 - NEW: distance object to target
            joint_limits,             # 6
            ee_vel * 0.1,            # 3 (scaled)
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
        try:
            tape_roll_xpos = self.data.body("tape_roll").xpos
            ee_finger_xpos = self.data.body("ee_finger").xpos
            target_position = np.array([0.7, 0.2, -0.1175])
            
            object_to_target_error = np.linalg.norm(tape_roll_xpos - target_position)
            ee_to_object_error = np.linalg.norm(ee_finger_xpos - tape_roll_xpos)
            
            # Primary reward: still object-to-target (for final goal)
            primary_reward = -1.0 * object_to_target_error
            
            # Dense approaching reward (this will dominate early learning)
            approach_reward = -10.0 * ee_to_object_error  # 10x stronger signal
            
            # Contact bonus (when robot gets close)
            contact_bonus = 0.0
            if ee_to_object_error < 0.2:  # Within 20cm
                contact_bonus = 50.0 * (0.2 - ee_to_object_error)  # Bonus increases as robot gets closer
            
            # Success bonus  
            success_bonus = 0.0
            if object_to_target_error < 0.05:
                success_bonus = 1000.0
            
            total_reward = primary_reward + approach_reward + contact_bonus + success_bonus
            
            return total_reward
            
        except Exception as e:
            print(f"Reward calculation error: {e}")
            return 0.0
