import numpy as np
import mujoco

def quat_distance(q1, q2):
    """Calculate distance between two quaternions"""
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Euclidean norms (quaternions q and -q represent same rotation)
    d1 = np.linalg.norm(q1 - q2)
    d2 = np.linalg.norm(q1 + q2)
    return min(d1, d2)

def quaternion_error(q_desired, q_current):
    """Calculate quaternion error as axis-angle vector"""
    # Normalize quaternions
    q_desired = np.array(q_desired, dtype=np.float64)
    q_current = np.array(q_current, dtype=np.float64)
    
    q_desired = q_desired / np.linalg.norm(q_desired)
    q_current = q_current / np.linalg.norm(q_current)
    
    # Calculate relative quaternion: q_error = q_desired * q_current^(-1)
    # q_current^(-1) = [w, -x, -y, -z] for unit quaternions
    q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    
    # Quaternion multiplication: q_desired * q_current_inv
    w1, x1, y1, z1 = q_desired
    w2, x2, y2, z2 = q_current_inv
    
    q_error = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])
    
    # Convert to axis-angle representation
    # For small angles, axis-angle ≈ 2 * [x, y, z] components
    if abs(q_error[0]) < 0.999:  # Not at identity
        # Normalize the vector part and scale by angle
        angle = 2 * np.arccos(np.clip(abs(q_error[0]), 0, 1))
        if np.sin(angle/2) > 1e-6:
            axis = q_error[1:4] / np.sin(angle/2)
            error_vector = axis * angle
        else:
            error_vector = q_error[1:4] * 2  # Small angle approximation
    else:
        error_vector = np.zeros(3)
    
    return error_vector

class GradientDescentIK:
    def __init__(self, model, data, step_size=0.1, tol=0.01, alpha=0.5, max_iter=100):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.max_iter = max_iter
        
        # Jacobians: MuJoCo fills first 3 rows of each
        self.jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        self.jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian

    def check_joint_limits(self, q):
        """Check if the joints are within limits"""
        q_limited = q.copy()
        for i in range(min(len(q), self.model.njnt)):
            q_min = self.model.jnt_range[i][0]
            q_max = self.model.jnt_range[i][1]
            q_limited[i] = np.clip(q[i], q_min, q_max)
        return q_limited

    def calculate(self, goal_pos, init_q, body_id, goal_quat):
        """Calculate desired joint angles for goal position and orientation"""
        
        if isinstance(goal_pos, np.ndarray) and len(goal_pos) == 6:
            # If goal is joint angles (legacy), just return them
            return goal_pos
        
        # Convert goal_pos to numpy array if it's not
        if not isinstance(goal_pos, np.ndarray):
            goal_pos = np.array(goal_pos)
        
        # Ensure goal_pos is 3D position
        if len(goal_pos) != 3:
            print(f"Warning: goal_pos should be 3D position, got shape {goal_pos.shape}")
            return init_q[:6].copy()
        
        # Normalize goal quaternion
        goal_quat = np.array(goal_quat, dtype=np.float64)
        goal_quat = goal_quat / np.linalg.norm(goal_quat)
        
        # Initialize joint angles
        q_current = init_q[:6].copy()
        
        # Store original state
        qpos_original = self.data.qpos.copy()
        qvel_original = self.data.qvel.copy()
        
        for iteration in range(self.max_iter):
            try:
                # Set temporary state for forward kinematics
                qpos_temp = self.data.qpos.copy()
                qpos_temp[:6] = q_current
                qvel_temp = np.zeros_like(self.data.qvel)
                
                self.data.qpos[:] = qpos_temp
                self.data.qvel[:] = qvel_temp
                mujoco.mj_forward(self.model, self.data)
                
                # Get current end-effector pose
                current_pos = self.data.body(body_id).xpos.copy()
                current_quat = self.data.body(body_id).xquat.copy()  # [w, x, y, z]
                
                # Calculate errors
                pos_error = goal_pos - current_pos
                rot_error = quaternion_error(goal_quat, current_quat)
                
                # Combine into 6-DOF error vector
                full_error = np.concatenate([pos_error, rot_error])
                
                # Check convergence
                pos_err_norm = np.linalg.norm(pos_error)
                rot_err_norm = np.linalg.norm(rot_error)
                
                if pos_err_norm < self.tol and rot_err_norm < 0.1:  # 0.1 rad ≈ 5.7 degrees
                    break
                
                # Compute Jacobians
                mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, body_id)
                
                # Build full 6-DOF Jacobian (6 x 6 for UR5)
                J_pos = self.jacp[:, :6]  # Position part (3 x 6)
                J_rot = self.jacr[:, :6]  # Rotation part (3 x 6)
                J_full = np.vstack([J_pos, J_rot])  # Full Jacobian (6 x 6)
                
                # Solve for joint velocity using pseudoinverse
                try:
                    # Check condition number to avoid singular configurations
                    if np.linalg.cond(J_full) < 1e6:
                        J_inv = np.linalg.pinv(J_full)
                    else:
                        # Use damped least squares for near-singular cases
                        damping = 0.01
                        J_inv = J_full.T @ np.linalg.inv(J_full @ J_full.T + damping * np.eye(6))
                    
                    # Calculate joint angle update
                    dq = self.alpha * J_inv @ full_error
                    
                    # Update joint angles
                    q_current += self.step_size * dq
                    
                    # Apply joint limits
                    q_current = self.check_joint_limits(q_current)
                    
                except np.linalg.LinAlgError:
                    print(f"Jacobian inversion failed at iteration {iteration}")
                    break
                    
            except Exception as e:
                print(f"IK iteration {iteration} failed: {e}")
                break
        
        # Restore original state
        self.data.qpos[:] = qpos_original
        self.data.qvel[:] = qvel_original
        mujoco.mj_forward(self.model, self.data)
        
        return q_current


# # Gradient Descent method
# import numpy as np 
# import mujoco 

# def quat_distance(q1, q2):
#     q1 = np.array(q1, dtype=np.float64)
#     q2 = np.array(q2, dtype=np.float64)
    
#     # Euclidean norms
#     d1 = np.linalg.norm(q1 - q2)
#     d2 = np.linalg.norm(q1 + q2)
    
#     return min(d1, d2)


# class GradientDescentIK:
#     def __init__(self, model, data, step_size, tol, alpha):
#         self.model = model
#         self.data = data
#         self.step_size = step_size
#         self.tol = tol
#         self.alpha = alpha
        
#         self.jacp = np.zeros((6, self.model.nv))
#         self.jacr = np.zeros((6, self.model.nv))
    
#     def check_joint_limits(self, q):
#         """Check if the joints is under or above its limits"""
#         for i in range(len(q)):
#             q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

#     #Gradient Descent pseudocode implementation
#     def calculate(self, goal, init_q, body_id, goal_rot):
#         """Calculate the desire joints angles for goal"""
#         self.data.qpos = init_q
#         mujoco.mj_forward(self.model, self.data)
     
#         mujoco.mj_jac(self.model, self.data, self.jacp, 
#             self.jacr, goal, body_id)
        
#         joint_vel = self.data.qvel()
#         ee_vel_rot = joint_vel @ self.jacr
#         ee_vel_pos = joint_vel @ self.jacp

#         # debugger 
#         import ipdb 
#         ipdb.set_trace()
#         error = goal - ee_vel_pos
#         error_rot = goal_rot - ee_vel_rot


#         while (np.linalg.norm(error) >= self.tol):
#             #calculate jacobian
#             mujoco.mj_jac(self.model, self.data, self.jacp, 
#                           self.jacr, goal, body_id)
            
#             #calculate gradient
#             grad = self.alpha * self.jacp.T @ error

#             #compute next step
#             self.data.qpos += self.step_size * grad

#             #check joint limits
#             self.check_joint_limits(self.data.qpos)
#             #compute forward kinematics
#             mujoco.mj_forward(self.model, self.data) 
#             #calculate new error
#             error = np.subtract(goal, self.data.body(body_id).xpos) 