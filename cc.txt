
class CartesianController(object):
    """
    ADAPTED FROM: https://github.com/roamlab/roam_bimm/blob/master/single_arm_teleoperation/teleop/src/control/cartesian_control.py

    This class performs cartesian control. I.e. turn EE desired positions into joint velocities. To initialize, it takes as input a robot URDF loaded from the parameter server.
    This class is in this file as opposed to `controller_optimizer.py` since this is a lower-level controller running underneath the controllers in that file and it is only used for simulation.
    """

    def __init__(
        self, mj_model, mj_data, max_vel=[0.1, 0.1], gains=[1.0, 1.0]  # kinematics,
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
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
        # these values are taken exactly from the urdf file
        self.urdf_rpy_rotations_to_check = [
            [3.1415926535897932, 0.0, 0.0],
            [0.0, 3.1415926535897932, 0.0],
            [0.0, 0.0, 3.1415926535897932],
            [1.5707963267948966, 0.0, 0.0],
            [0.0, 1.5707963267948966, 0.0],
            [0.0, 0.0, 1.5707963267948966],
            [0.7853981633974483, 0.0, 0.0],
            [0.0, 0.7853981633974483, 0.0],
            [0.0, 0.0, 0.7853981633974483],
        ]
        self.quats_from_urdf_rpy = [
            Rotation.from_euler("xyz", rpy).as_quat(scalar_first=True)
            for rpy in self.urdf_rpy_rotations_to_check
        ]
        # only for printing purposes below (can delete) >>>
        self.quat_from_urdf_rpy_x = Rotation.from_euler(
            "xyz", [1.5707963267948966, 0.0, 0.0]
        ).as_quat(scalar_first=True)
        self.quat_from_urdf_rpy_y = Rotation.from_euler(
            "xyz", [0.0, 1.5707963267948966, 0.0]
        ).as_quat(scalar_first=True)
        self.quat_from_urdf_rpy_z = Rotation.from_euler(
            "xyz", [0.0, 0.0, 1.5707963267948966]
        ).as_quat(scalar_first=True)

    def _replace_quat_with_urdf_precision(self, quat_to_check):
        """Custom code for UR5 to fix inconsistency between URDF file and mujoco (see README.md). If quaternion from mujoco is close to quaternion obtained from raw URDF euler angle data, then keep the latter quat (which will be slightly different)."""
        for quat_from_urdf_rpy in self.quats_from_urdf_rpy:
            if np.allclose(quat_to_check, quat_from_urdf_rpy):
                return quat_from_urdf_rpy
        return quat_to_check

    def forward_kinematics(self, joint_values):
        """Take as input current joint values and gives as output array of 4x4 transforms from base to each link of the robot and the EE.

        Assumptions:
        mujoco file where robot is a chain where each body has exactly one joint with one child body and so forth. (otherwise would need to iterate over bodies and not over joints)
        """
        transforms_from_base = []
        T = np.eye(4, 4)  # base to joint

        """ joint_values = [0, -1.570796327, 0, -1.570796327, 0, 0]
        joint_values = [
            -0.03438299,
            -1.45211394,
            1.48614786,
            4.64798633,
            -1.56468767,
            1.40254659,
        ] """
        """ printa()
        printa("joint values", joint_values) """
        for i_joint in range(6):
            # T[np.abs(T) < 1e-7] = 0.0
            printk(
                "\n JOINT",
                i_joint,
                # mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i_joint),
                # dir(self.mj_model.joint(i_joint)),
                # dir(self.mj_data.joint(i_joint)),
            )
            """ printa(
                "  body ",
                self.mj_model.joint(i_joint).bodyid,
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).name,
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat,
            ) """
            prev_T_curr = np.eye(4, 4)
            prev_T_curr[:3, 3] = self.mj_model.body(
                self.mj_model.joint(i_joint).bodyid
            ).pos
            printk(
                type(self.mj_model.body(self.mj_model.joint(i_joint).bodyid).pos),
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).pos.dtype,
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).pos,
            )
            printk("prev_T_curr translation\n", prev_T_curr, 6)

            # static part (link)
            rot = np.eye(4, 4)
            """ printa(
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid),
                # self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat,
                dir(self.mj_model.body(self.mj_model.joint(i_joint).bodyid)),
            ) """
            quat = self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat
            # Custom code to fix inconsistency between URDF files and mujoco (see README). If quaternion from mujoco is close to quat obtained from raw URDF euler angle data, then keep the latter quat (which will be slightly different)
            printk(
                "quat, quat_from_urdf_rpy, all_close",
                quat,
                self.quat_from_urdf_rpy_y,
                np.allclose(quat, self.quat_from_urdf_rpy_y),
            )
            quat = self._replace_quat_with_urdf_precision(quat)
            """ if np.allclose(quat, self.quat_from_urdf_rpy):
                quat = self.quat_from_urdf_rpy """

            rot[:3, :3] = Rotation.from_quat(
                quat,
                scalar_first=True,
            ).as_matrix()
            printk(
                "quat",
                type(self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat),
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat.dtype,
                self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat,
                Rotation.from_euler("xyz", [0.0, 1.5707963267948966, 0.0]).as_quat(
                    scalar_first=True
                ),
            )
            """ rot[:3, :3] = Rotation.from_quat(
                Rotation.from_euler("xyz", [0.0, 1.5707963267948966, 0.0]).as_quat(
                    scalar_first=True
                ),
                scalar_first=True,
            ).as_matrix() """  # TODO: delete this line
            # exit()
            printk(
                "prev_T_curr rot\n",
                # self.mj_model.body(self.mj_model.joint(i_joint).bodyid).quat,
                rot,
            )

            prev_T_curr = np.dot(prev_T_curr, rot)
            printk("prev_T_curr trans and rot\n", prev_T_curr)

            T = np.dot(T, prev_T_curr)
            # T[np.abs(T) < 1e-7] = 0.0
            printk("T before\n", T)
            transforms_from_base.append(T.copy())

            # dynamic part (joint)
            assert (
                self.mj_model.joint(i_joint).type == 3
            ), f"Wrong joint type: {self.mj_model.joint(i_joint).type}"
            T_rot = np.eye(4, 4)
            """ printa(
                "axis, val, norm",
                self.mj_model.joint(i_joint).axis,
                joint_values[i_joint],
                np.linalg.norm(
                    self.mj_model.joint(i_joint).axis * joint_values[i_joint]
                ),
            ) """
            T_rot[:3, :3] = Rotation.from_rotvec(
                self.mj_model.joint(i_joint).axis * joint_values[i_joint]
            ).as_matrix()
            printk(
                "axis",
                self.mj_model.joint(i_joint).axis.dtype,
                self.mj_model.joint(i_joint).axis,
            )
            printk("T_rot\n", T_rot)
            T = np.dot(T, T_rot)
            # T[np.abs(T) < 1e-7] = 0.0
            printk("T after\n", T)

        # last link (EE box) is attached to wrist via fixed joint that mujoco ignores so need to go body-to-body directly
        prev_T_curr = np.eye(4, 4)
        last_bodyid = self.mj_model.joint(i_joint).bodyid + 1
        prev_T_curr[:3, 3] = self.mj_model.body(last_bodyid).pos
        printk("prev_T_curr translation\n", prev_T_curr)

        rot = np.eye(4, 4)
        """ printk(
            self.mj_model.body(last_bodyid),
        ) """
        quat = self.mj_model.body(last_bodyid).quat
        # Custom code to fix inconsistency between URDF files and mujoco (see README). If quaternion from mujoco is close to quat obtained from raw URDF euler angle data, then keep the latter quat (which will be slightly different)
        printk(
            "quat, quat_from_urdf_rpy, all_close",
            quat,
            self.quat_from_urdf_rpy_y,
            np.allclose(quat, self.quat_from_urdf_rpy_y),
        )
        quat = self._replace_quat_with_urdf_precision(quat)
        rot[:3, :3] = Rotation.from_quat(
            quat,
            scalar_first=True,
        ).as_matrix()
        printk("prev_T_curr rot\n", quat, rot)

        prev_T_curr = np.dot(prev_T_curr, rot)
        printk("prev_T_curr trans and rot\n", prev_T_curr)

        T = np.dot(T, prev_T_curr)
        # T[np.abs(T) < 1e-7] = 0.0

        printk("T final\n", T)

        return transforms_from_base, T  # bTee

    def rotation_from_matrix(self, matrix):
        """
        This function will return the angle-axis representation of the rotation
        contained in the input matrix.

        Usage: angle, axis = rotation_from_matrix(R).

        NOTE: only used for debugging. can remove and replace with scipy.Rotation.as_rotationve()
        """

        R = np.array(matrix, dtype=np.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = np.linalg.eig(R33.T)
        i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = np.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = np.linalg.eig(R)
        i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (np.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa - 1.0) * axis[0] * axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa - 1.0) * axis[0] * axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa - 1.0) * axis[1] * axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    def get_jacobian(self, b_T_ee, joint_transforms):
        """
        This function will assemble the jacobian of the robot using the
        current joint transforms and the transform from the base to the end
        effector (b_T_ee).

        Usage: J = self.get_jacobian(b_T_ee, joint_transforms)
        """

        J = np.zeros((6, 6))
        for i_joint in range(0, len(joint_transforms)):
            # rigid body transfer of joint rotation to end-effector motion
            b_T_j = joint_transforms[i_joint]
            j_T_ee = np.dot(np.linalg.inv(b_T_j), b_T_ee)
            ee_T_j = np.linalg.inv(j_T_ee)
            ee_R_j = ee_T_j[0:3, 0:3]
            j_trans_ee = j_T_ee[0:3, 3]
            S = np.zeros((3, 3))
            S[0, 1] = -j_trans_ee[2]
            S[0, 2] = j_trans_ee[1]
            S[1, 0] = j_trans_ee[2]
            S[1, 2] = -j_trans_ee[0]
            S[2, 0] = -j_trans_ee[1]
            S[2, 1] = j_trans_ee[0]
            RS = np.dot(ee_R_j, S)
            # choose the right column to put into Jacobian
            # J[0:3, i] = -np.dot(RS, self.joint_axes[i])
            J[0:3, i_joint] = -np.dot(RS, self.mj_model.joint(i_joint).axis)
            J[3:6, i_joint] = np.dot(ee_R_j, self.mj_model.joint(i_joint).axis)
        return J

    def cartesian_command(self, q_current, x_command: Command):
        """This is the callback which will be executed when the cartesian control
        recieves a new command. At the end of this callback,
        you should publish to the /joint_velocities topic.

        Input: desired end effector position and current joint values.
        Output: joint velocities to reach desired end effector position"""

        # Perform forward kinematics to obtain current end effector pose:
        """ q_current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # q_current = [0.000, -1.343, 2.002, -2.228, -1.571, 1.571]
        x_command = Command(1.0, 0.0, 0.0) """
        # TODO: deactivate these << 3 lines (used to comapre mjcf to urdf in a canonical setup)
        joint_transforms, b_T_ee = self.forward_kinematics(q_current)
        # b_T_ee is different compared to ROS setup when loading mjcf/xml file (as opposed to urdf file where error doesnt occur) in that some 0's have sign flipped. This propagates down to an actually different dq, which is a problem!
        """ for i, transform in enumerate(joint_transforms):
            printk(i, "\n", transform.round(4))
        printk(
            "Kinematics",
            len(joint_transforms),
            "\n",
            q_current,
            x_command,
            "b_T_ee*\n",
            b_T_ee,
        ) """

        """ for i, transform in enumerate(joint_transforms):
            printa(i)
            printa(np.round(transform, 3)) """
        # printa(np.round(b_T_ee, 3))

        # Obtain desired pose from command:
        """ trans = tf.transformations.translation_matrix(
            (command.translation.x, command.translation.y, command.translation.z)
        ) """
        trans = np.eye(4)
        """ printa("x_command", x_command) """
        trans[:3, 3] = np.array(
            [
                x_command.trans_x,
                x_command.trans_y,
                x_command.trans_z,
            ]
        )
        """ printk("desired, trans\n", trans.round(4)) """

        """ rot = tf.transformations.quaternion_matrix(
            (
                command.rotation.x,
                command.rotation.y,
                command.rotation.z,
                command.rotation.w,
            )
        ) """
        rot = np.eye(4, 4)
        rot[:3, :3] = Rotation.from_quat(
            np.array(
                [
                    x_command.rot_x,
                    x_command.rot_y,
                    x_command.rot_z,
                    x_command.rot_w,
                ]
            )
        ).as_matrix()
        """ printk("desired, rot\n", rot.round(4)) """
        b_T_des = np.dot(trans, rot)
        """ printk("b_T_des desired trans@rot\n", b_T_des.round(4)) """

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
        # dx_e += gain * velocity error
        # velocity error =
        """ printk("dx_e 2\n", dx_e.round(4)) """

        # Normalize to obtain max end-effector velocity of 0.1m/s
        if np.linalg.norm(dx_e) > self.max_vel_trans:
            dx_e = (self.max_vel_trans / np.linalg.norm(dx_e)) * dx_e
            """ printk("dx_e 3\n", dx_e.round(4)) """

        # Get desired angular velocity in local frame
        # c_R_des = c_T_des[0:3, 0:3]
        ee_R_des = ee_T_des[:3, :3]
        """ printk("ee_R_des\n", ee_R_des.round(4)) """
        ee_rot_des = Rotation.from_matrix(ee_R_des)
        rotationvec = ee_rot_des.as_rotvec()
        """ printk("rotationvec", rotationvec) """
        angle = np.linalg.norm(rotationvec)
        axis = rotationvec / angle
        """ printk("angle, axis", angle, axis) """
        angle, axis = self.rotation_from_matrix(ee_R_des)
        """ printk("angle, axis", angle, axis) """
        # angle, axis = self.kinematics.rotation_from_matrix(c_R_des)
        dw_e = angle * axis
        """ printk("dw_e 1\n", dw_e.round(4)) """

        dw_e *= self.rot_gain
        """ printk("dw_e 2\n", dw_e.round(4)) """

        # Normalize to max end-effector angular velocity of 1 rad/s
        if np.linalg.norm(dw_e) > self.max_vel_ang:
            dw_e = (self.max_vel_ang / np.linalg.norm(dw_e)) * dw_e
            """ printk("dw_e 3\n", dw_e.round(4)) """

        # Assemble twist (translational and angular velocities vector)
        dv_e = np.zeros(6)
        dv_e[0:3] = dx_e
        dv_e[3:6] = dw_e
        """ printk("dv_e 3\n", dv_e.round(4)) """

        # Convert to joint velocities with jacobian pseudo-inverse
        J = self.get_jacobian(b_T_ee, joint_transforms)
        """ printk("J\n", J.round(4), J.shape) """
        Jps = np.linalg.pinv(J, 1.0e-2)
        """ printk("J pseudo inverse \n", Jps.round(4), Jps.shape) """
        dq = np.dot(Jps, dv_e)
        # dq[-3:] = 0.0  # TODO: deactivate this when it works as it should
        printa("dq", np.round(dq, 4), dq.shape)

        # exit()

        return dq  # Return commanded joint velocities

    def set_desired_ee_pos(self, x_command: Command):
        self.desired_ee_pos = x_command

