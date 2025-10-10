import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
# for reward structure 
from dataclasses import dataclass, asdict

# import cc_with_cmd as cc  # Cartesian controller
import cartesian_controller as cc 
from scipy.spatial.transform import Rotation 

gym.register(
    id="UR5-v1",
    entry_point="ur5_push_env:ur5",
    max_episode_steps=500,
)

class ur5(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path="./env/assets/scene.xml", frame_skip=10, fix_orientation = True, **kwargs):
        super().__init__(
            model_path,
            frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32),  # 3 more for cvel of tape roll + 1 for contact
            **kwargs,
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[3] = np.pi
        self.init_qvel = self.data.qvel.copy()

        # joint action scaling (used as velocity limits)
        self.act_mid = np.zeros(6)
        self.act_rng = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])

        # action: [dx, dy, dz, d_rx, d_ry, d_rz]
        self.fix_orientation = fix_orientation
        if self.fix_orientation:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)

        # scaling 
        self.max_delta_pos = 0.02 # meters per step 
        self.max_delta_rot = 0.02    # radians per step

        # targets
        self.ee_target_pos = self.data.body("ee_finger").xpos.copy()
        self.ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        self.target_position = np.array([0.7, 0.2, -0.1175])
        self.prev_tape_roll_pos = None

        downward_rotation = Rotation.from_euler('xyz', [-np.pi, 0, -np.pi])
        self.fixed_down_quat = downward_rotation.as_quat(scalar_first = True)
        self.q_des = Rotation.from_quat(self.fixed_down_quat, scalar_first=True)

        # controller randomly sets state to discontinuous position/time 
        # you can't set the state of the word how the controller does it 
        # black box analogy where only policy can impact the environment since the policy is the thing that does commands based on controller which is black box
        # data_copy = copy.deepcopy(self.data)
        self.cartesian_controller = cc.CartesianController(self.model, self.data)
        self.tape_roll = self.data.body("tape_roll")
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


        # print(self.ee_target_quat, self.ee_finger.xquat.copy())
        # Convert back to quaternion and normalize
        if self.fix_orientation:
            self.ee_target_quat = self.fixed_down_quat
        else:
            # rotation update
            delta_rot = action[3:] * self.max_delta_rot
            
            # Create rotation from delta rotation vector
            delta_rotation = Rotation.from_rotvec(delta_rot)
            
            # Create rotation from current quaternion
            current_rotation = Rotation.from_quat(self.ee_finger.xquat.copy(), scalar_first=True)
            # Compose rotations
            new_rotation = current_rotation * delta_rotation

            self.ee_target_quat = self._normalize_quaternion(
                new_rotation.as_quat(scalar_first=True)
            )
            
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


        for i in range(self.frame_skip):
            q_current = self.data.qpos[:6].copy()
            dq = self.cartesian_controller.cartesian_command(q_current, cmd)
            dq = np.clip(dq, -self.act_rng, self.act_rng)
            self.do_simulation(dq, 1)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward_dict = self.reward_dict() 
        reward = self.get_reward()

        # only step should change the actual env 
        self.prev_tape_roll_pos = self.tape_roll.xpos
        self.prev_ee_to_obj = np.linalg.norm(self.ee_finger.xpos - self.tape_roll.xpos)

        tape_roll_xpos = self.data.body("tape_roll").xpos
        target_position = self.target_position
        object_to_target_error = np.linalg.norm(tape_roll_xpos - target_position)

        terminated = object_to_target_error < 0.05 # within 5 cm 
        truncated = np.linalg.norm(self.ee_finger.xpos - tape_roll_xpos) > 0.8
        if truncated:
            reward -= 100

        return obs, reward, terminated, truncated, reward_dict

    # def _get_obs(self):
    #     ee = self.ee_finger 
    #     obj = self.tape_roll

    #     ee_pos = ee.xpos  # views, no .copy()
    #     # self.data.body("ee_finger").xpos is already a NumPy view into MuJoCo’s C-struct memory?
    #     obj_pos = obj.xpos
    #     obj_vel = obj.cvel[3:]          # linear vel (3,)
    #     to_obj = obj_pos - ee_pos       # (3,)
    #     to_goal = self.target_position - obj_pos

    #     # normalized direction to goal
    #     d = to_goal / (np.linalg.norm(to_goal) + 1e-8)
    #     obj_speed_toward_goal = np.dot(obj_vel, d)

    #     ee_vel = ee.cvel[3:]
    #     ee_speed_toward_obj = np.dot(ee_vel, (to_obj / (np.linalg.norm(to_obj)+1e-8)))

    #     obs = np.concatenate([
    #         to_obj,                 # 3
    #         to_goal,                # 3
    #         obj_vel,                # 3 (or drop to keep very small)
    #         np.array([obj_speed_toward_goal], np.float32),  # 1
    #         np.array([ee_speed_toward_obj], np.float32),    # 1
    #         np.array([ee_pos[2]], np.float32),              # 1
    #         np.array([float(self.tape_roll_cont("ee_finger"))], np.float32)  # 1
    #     ], dtype=np.float32)

    #     return obs
        
    def _get_obs(self):
        qpos_ur5 = self.data.qpos[:6]
        qvel_ur5 = self.data.qvel[:6]

        # in your observation you want .xpos and no .copy() since you want it at that specific state vs controllers
        ee_pos = self.data.body("ee_finger").xpos
        ee_quat = self.ee_finger.xquat
        tape_roll_pos = self.data.body("tape_roll").xpos
        # possibly add tape roll orientation

        ee_to_object = tape_roll_pos - ee_pos
        object_to_target = self.target_position - tape_roll_pos

        ee_vel = self.ee_finger.cvel[3:]
        tape_roll_vel = self.data.body("tape_roll").cvel[3:]

        obs = np.concatenate(
            [
                qpos_ur5,
                qvel_ur5 * 0.1,
                ee_pos,
                ee_quat,
                tape_roll_pos,
                self.target_position,
                ee_to_object,
                object_to_target,
                ee_vel * 0.1,
                tape_roll_vel,
                np.array([float(self.tape_roll_cont("ee_finger"))], dtype=np.float32)
            ]
        )
        return obs.astype(np.float32)
    
    def tape_roll_cont(self, geom: str):
        tape_id = self.tape_roll.id
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            # If tape roll touches something other than the allowed geom(s)
            if (self.model.geom_bodyid[g1] == tape_id and g2 == geom_id) \
            or (self.model.geom_bodyid[g2] == tape_id and g1 == geom_id):
                return True
            
        return False

    def reset_random_pose(self, max_iters=1000, tol=1e-5):
        qpos = self.init_qpos.copy()

        # calculated based on the xml 
        # target_pos = np.array([np.random.uniform(0.202, 0.802),  # within table X
        #               np.random.uniform(-0.6, 0.6),     # within table Y  
        #               np.random.uniform(-0.1475, -0.05) # above surface
        #               ])

        target_pos = np.array([np.random.uniform(0.3, 0.6),  # within table X
                      np.random.uniform(-0.1, 0.5),     # within table Y  
                      np.random.uniform(-0.1475, -0.05) # above surface
                      ])
        
        # np.array([1, 0, 0.0, 0]) # xyzw 
        # safe start based on XML
        qpos_1 = np.array([0, -0.44, 1.01, -0.44, -1.38, 0])
        qpos_2 = np.array([-0.754, -0.628, 1.95, 0, 0.0232, 0])
        q_pos_fix = np.array([0.126, -0.942, 1.95, -2.58, -1.63, 0])
        # q_pos_fix = np.array([0.126, -1.51, 2.26, -2.26, -1.63, 0])

        if self.fix_orientation: 
            qpos[:6] = q_pos_fix
            self.set_state(qpos, self.data.qvel)
        
        else:
            if np.random.random() < 0.3:  
                qpos[:6] = qpos_2   
                self.set_state(qpos, self.data.qvel)

            else:
                qpos[:6] = qpos_1
                self.set_state(qpos, self.data.qvel)
                target_quat = self.ee_finger.xquat.copy()
                # mujoco.mj_setState(self.model, self.data, qpos, mujoco.mjtState.mjSTATE_QPOS) # this doesnt call forward hence quat = 0 norm 

                for _ in range(max_iters):
                    x_command = cc.Command(*target_pos, *target_quat) # takes an iterable and passes it as arguments in the order
                    qvel = np.clip(self.cartesian_controller.cartesian_command(self.data.qpos, x_command), -3.15, 3.15)
                    self.do_simulation(qvel, 1) # the controller is a velocity controller so you can just pass in qvel

                    error = np.linalg.norm(self.ee_finger.xpos - target_pos)
                    if error < tol:
                        break 


    def reset_model(self):
        # from the sim
        #base_pose= np.array([0, -0.44, 1.01, -0.44, -1.38, 0])
        # add gaussian noise 
        # noise_std = 0.2 
        # joint_noise = np.random.normal(0, noise_std, 6)

        self.reset_random_pose()
        qpos = self.data.qpos.copy()

        qpos[9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # reset object upright
        
        # randomizing the tape roll position
        new_qpos = np.random.randn(2) * 0.05 + qpos[6:8] 
        qpos[6:8] = new_qpos

        self.set_state(qpos, self.data.qvel)

        if not self.tape_roll_cont("table") and not self.fix_orientation:
            self.reset_model()

        self.ee_target_pos = self.ee_finger.xpos.copy()
        self.ee_target_quat = self.ee_finger.xquat.copy()

        # reset prev tape roll position for reward shaping
        self.prev_tape_roll_pos = self.tape_roll.xpos.copy()
        self.prev_ee_to_obj = np.linalg.norm(self.ee_finger.xpos - self.tape_roll.xpos)


        return self._get_obs()
    
    @dataclass
    class reward_scales:
        ee_to_object: float = -0.10
        obj_to_target: float = -0.10
        progress: float = 50.0
        contact: float = 10.0
        success: float = 500.0
        velocity_alignment: float = 10.0
        ee_approaching: float = 5.0
        orientation: float = -0.01
    
    
    def reward_dict(self) -> dict:
        s = self.reward_scales()

        tape_pos = self.tape_roll.xpos
        ee_pos = self.ee_finger.xpos
        obj_vel = self.tape_roll.cvel[3:]
        target_dir = self.target_position - tape_pos
        target_dir /= (np.linalg.norm(target_dir) + 1e-8)

        ee_to_obj = np.linalg.norm(ee_pos - tape_pos)
        obj_to_target = np.linalg.norm(self.target_position - tape_pos)

        # progress of object toward target
        prev_dist = np.linalg.norm(self.target_position - self.prev_tape_roll_pos)
        progress = prev_dist - obj_to_target

        # end-effector approaching object
        # ee_approach = self.prev_ee_to_obj - ee_to_obj

        contact = float(self.tape_roll_cont("ee_finger"))
        vel_align = np.clip(np.dot(obj_vel, target_dir), -0.1, 0.1)

        if self.fix_orientation:
            q_cur = Rotation.from_quat(self.ee_finger.xquat, scalar_first=True)
            ang_err = (self.q_des * q_cur.inv()).magnitude()
        else:
            ang_err = 0.0

        success = obj_to_target < 0.05

        return {
            # "ee_approach": 5.0 * ee_approach * (1.0 - contact),
            "ee_distance": - 10.0 * ee_to_obj * (1.0 - contact),
            "contact": 10.0 if contact else 0.0, 
            "contact_progress": 100.0 * progress * contact,
            "velocity_alignment": 10.0 * vel_align * contact,
            "success": 500.0 if success else 0.0,
            "orientation": -0.01 * ang_err,
        }

    def get_reward(self):
       return sum(self.reward_dict().values())       


# def reward_dict(self) -> dictd:
    #     s = self.reward_scales()

    #     tape_pos = self.data.body("tape_roll").xpos
    #     ee_pos = self.data.body("ee_finger").xpos
        
    #     # Compute raw values
    #     ee_to_obj = np.linalg.norm(ee_pos - tape_pos)
    #     obj_to_target = np.linalg.norm(self.target_position - tape_pos)
        
    #     progress = 0.0
    #     if hasattr(self, "prev_tape_roll_pos"):
    #         prev_dist = np.linalg.norm(self.target_position - self.prev_tape_roll_pos)
    #         progress = prev_dist - obj_to_target
    #     self.prev_tape_roll_pos = tape_pos.copy()
        
    #     obj_vel = self.data.body("tape_roll").cvel[3:]
    #     target_dir = self.target_position - tape_pos
    #     target_dir /= (np.linalg.norm(target_dir) + 1e-8)
    #     vel_align = np.dot(obj_vel, target_dir)
        
    #     ang_err = 0.0
    #     if self.fix_orientation:
    #         q_cur = Rotation.from_quat(self.ee_finger.xquat, scalar_first=True)
    #         ang_err = (self.q_des * q_cur.inv()).magnitude()
        
    #     # Apply scales and return
    #     return {
    #         "ee_to_object": s.ee_to_object * ee_to_obj**2,
    #         "obj_to_target": s.obj_to_target * obj_to_target**2,
    #         "progress": s.progress * progress,
    #         "contact": s.contact if self.tape_roll_cont("ee_finger") else 0.0,
    #         "success": s.success if obj_to_target < 0.05 else 0.0,
    #         "velocity_alignment": s.velocity_alignment * vel_align,
    #         "orientation": s.orientation * ang_err,
    #     }



