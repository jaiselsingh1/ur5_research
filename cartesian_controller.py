import numpy as np
import math
from scipy.spatial.transform import Rotation
import typing
import mujoco as mj 


class Command(typing.NamedTuple):
    """Desired EE pose"""
    trans_x: float = 0.0
    trans_y: float = 0.0
    trans_z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0
    rot_w: float = 1.0
        

    def __str__(self):
        p = 3
        return f"{round(self.trans_x,p)}, {round(self.trans_y,p)}, {round(self.trans_z,p)}, {round(self.rot_x,p)}, {round(self.rot_y,p)}, {round(self.rot_z,p)}, {round(self.rot_w,p)}"
        return f"{np.round(self.trans_x,p)}, {np.round(self.trans_y,p)}, {np.round(self.trans_z,p)}, {np.round(self.rot_x,p)}, {np.round(self.rot_y,p)}, {np.round(self.rot_z,p)}, {np.round(self.rot_w,p)}"

class CartesianController(object):
    """
    ADAPTED FROM: https://github.com/roamlab/roam_bimm/blob/master/single_arm_teleoperation/teleop/src/control/cartesian_control.py

    This class performs cartesian control. I.e. turn EE desired positions into joint velocities. To initialize, it takes as input a robot URDF loaded from the parameter server.
    This class is in this file as opposed to `controller_optimizer.py` since this is a lower-level controller running underneath the controllers in that file and it is only used for simulation.
    """

    def __init__(
        self, mj_model: mj.MjModel, mj_data: mj.MjData, max_vel=[0.5, 1.0], gains=[1.0, 5.0]  # kinematics, 
    ):
        self.model = mj_model
        self.data = mj_data

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.ee_body = self.data.body("ee_finger")
       
        # Create kinematics object for robot urdf:
        # self.kinematics = kinematics

        # Set max end effector velocity:
        # defaults in function signature are from cartesian_control.py
        # max_vel = [0.5, 1.0]  # robot_controller.py default
        self.max_vel_trans = max_vel[0]  # m/s
        self.max_vel_ang = max_vel[1]  # rad/s

        # Cartesian controller velocity gains:
        # gains = [4.0, 2.0]  # ur5_trakstar_real.launch
        # gains = [4.0, 1.0]  # robot_controller.py default
        self.trans_gain = gains[0]
        self.rot_gain = gains[1]

    def homogenous_matrix(self, x, y, z, rot_x, rot_y, rot_z, rot_w):
        A_from_B = np.eye(4)
        A_from_B[:3,3] = np.array(
            [
                x,
                y,
                z,
            ])

        A_from_B[:3, :3] = Rotation.from_quat(
            np.array(
                [
                    rot_x,
                    rot_y,
                    rot_z,
                    rot_w,
                ]
            )
        ).as_matrix()

        return A_from_B


    def cartesian_command(self, q_current, x_command: Command):
        """This is the callback which will be executed when the cartesian control
        recieves a new command. At the end of this callback,
        you should publish to the /joint_velocities topic.

        Input: desired end effector position and current joint values.
        Output: joint velocities to reach desired end effector position"""

        # mj.mj_setState(self.model, self.data, q_current, mj.mjtState.mjSTATE_QPOS)
        # mj.mj_forward(mj.model, mj.data)
        mj.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.ee_body.id)

        xquat = self.ee_body.xquat.copy()
        xpos = self.ee_body.xpos.copy()
        
        target_xpos = np.array([x_command.trans_x, x_command.trans_y, x_command.trans_z])
        target_xquat = np.array([x_command.rot_x, x_command.rot_y, x_command.rot_z, x_command.rot_w])

        R_current = Rotation.from_quat(xquat, scalar_first=True)
        # mujoco does w, x, y, z
        R_desired = Rotation.from_quat(target_xquat)

        R_diff = R_desired * R_current.inv()
        # diff in rot 
        max_ang_vel = self.max_vel_ang
        rvec = R_diff.as_rotvec()
        rvec_norm = np.linalg.norm(rvec)

        if rvec_norm > 1e-6:
            rvec = rvec / rvec_norm * np.clip(rvec_norm / self.model.opt.timestep, 0, max_ang_vel) * self.rot_gain
        else:
            rvec = np.zeros_like(rvec)


        # jac = np.concatenate([self.jacp[:, :6], self.jacr[:, :6]])
        # jac_pinv = np.linalg.pinv(jac, 1e-4)

         # convert difference in postions/rot into target vels 
        x_diff = target_xpos - xpos 
        dx_des = (x_diff / self.model.opt.timestep) * self.trans_gain

        xdot_des = np.concatenate([dx_des, rvec])

        # damped psuedo inverse
        jac = np.empty((6, 6))
        jac[:3, :] = self.jacp[:, :6]
        jac[3:, :] = self.jacr[:, :6]

        lam = 1e-3  # damping factor
        JT = jac.T
        A = JT @ jac + lam * np.eye(6)
        b = JT @ xdot_des
        qvel = np.linalg.solve(A, b)

        # qvel = jac_pinv @ xdot_des

        return qvel

    def set_desired_ee_pos(self, x_command: Command):
        self.desired_ee_pos = x_command
