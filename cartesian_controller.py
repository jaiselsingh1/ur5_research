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
        self, mj_model: mj.MjModel, mj_data: mj.MjData, max_vel=[0.1, 0.1], gains=[1.0, 1.0]  # kinematics,
    ):
        self.model = mj_model
        self.data = mj_data

        self.jacp = np.zeros(3, self.model.nv)
        self.jacr = np.zeros(3, self.model.nv)
        self.ee_body = self.model.body("ee_finger")
        # Create kinematics object for robot urdf:
        # self.kinematics = kinematics

        # Set max end effector velocity:
        # defaults in function signature are from cartesian_control.py
        max_vel = [0.5, 1.0]  # robot_controller.py default
        self.max_vel_trans = max_vel[0]  # m/s
        self.max_vel_ang = max_vel[1]  # rad/s

        # Cartesian controller velocity gains:
        gains = [4.0, 2.0]  # ur5_trakstar_real.launch
        gains = [4.0, 1.0]  # robot_controller.py default
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

        # Perform forward kinematics to obtain current end effector pose:
        joint_transforms, b_T_ee = 
        # b_T_ee is different compared to ROS setup when loading mjcf/xml file (as opposed to urdf file where error doesnt occur) in that some 0's have sign flipped. This propagates down to an actually different dq, which is a problem!
        mj.mj_setState(self.model, self.data, q_current, mj.mjtState.mjSTATE_QPOS)
        mj.mj_forward(mj.model, mj.data)
        mj.mj_jacBody(mj.model, mj.data, self.jacp, self.jacr, self.ee_body.id)

        # transformation matrix from end effector to world
        world_from_ee = homogenous_matrix()
        # give values from self.body (xpos, xquat) for ee finger

        # get a world from ee_desired from the command 


        # Obtain end effector pose delta to move towards command:
        # c_T_des = np.dot(tf.transformations.inverse_matrix(b_T_ee), b_T_des)
        """ printk("inv(b_T_ee)*\n", np.linalg.inv(b_T_ee).round(4)) """
        ee_T_des = np.dot(
            np.linalg.inv(b_T_ee), b_T_des
        )  # THIS IS DIFFERENT FROM tf.transformations.inverse_matrix(b_T_ee) in that some 0's have different signs
        """ printk("c_T_des* \n", ee_T_des.round(4)) """
        # ee_T_des[np.abs(ee_T_des) < 1e-9] = 0.0
        """ printk("c_T_des after* \n", ee_T_des.round(4)) """

        # Get desired translational velocity in local frame
        """ dx_e = np.zeros(3)
        dx_e[0] = c_T_des[0][3]
        dx_e[1] = c_T_des[1][3]
        dx_e[2] = c_T_des[2][3] """
        dx_e = ee_T_des[:3, 3]
        """ printk("dx_e 1\n", dx_e.round(4)) """

        dx_e *= self.trans_gain

        Jps = np.linalg.pinv(J, 1.0e-2)
        """ printk("J pseudo inverse \n", Jps.round(4), Jps.shape) """
        dq = np.dot(Jps, dv_e)
        # dq[-3:] = 0.0  # TODO: deactivate this when it works as it should
        printk("dq", np.round(dq, 4), dq.shape)

        # exit()

        return dq  # Return commanded joint velocities

    def set_desired_ee_pos(self, x_command: Command):
        self.desired_ee_pos = x_command
