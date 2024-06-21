import numpy as np
import pinocchio as pin
from pathlib import Path
from numpy.linalg import pinv


class SysIdentification():
    def __init__(self, urdf_filename, floating_base, viz=None):
        # Creat robot model and data
        self.__floating_base = floating_base
        if self.__floating_base:
            self.__rmodel = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())
        else:
            self.__rmodel = pin.buildModelFromUrdf(urdf_filename)
        self.__rdata = self.__rmodel.createData()
        
        # Set the gravity vector in pinocchio
        self.__rmodel.gravity.linear = np.array([0, 0, -9.81])
        
        # Dimensions of robot confgiuration space and velocity vector
        self.nq = self.__rmodel.nq
        self.nv = self.__rmodel.nv
        
        # Initialize the regressor matrix with proper dimension
        # For now only considering 10 inertial parameters for each link
        # [m h_x h_y h_z I_xx I_xy I_xz I_yy I_yz I_zz]
        self.__num_inertial_prameters = 10
        self.__num_links = self.__rmodel.njoints-1 # In pinocchio, universe is always in the kinematic tree with joint[id]=0
        self.__Y = np.zeros((self.nv, self.__num_inertial_prameters * self.__num_links), dtype=np.float32)
        
        # List of the end_effector names
        # TODO: Later put all changing parameters in a separate yaml config file
        self.__end_eff_frame_names = ["HL_ANKLE", "HR_ANKLE", "FL_ANKLE", "FR_ANKLE"]
        self.__endeff_ids = [
            self.__rmodel.getFrameId(name)
            for name in self.__end_eff_frame_names
        ]
        self.__nb_ee = len(self.__end_eff_frame_names)
        
        self.__init_motion_subspace_dict()
        self.__show_kinematic_tree()
    
    def __show_kinematic_tree(self):
        print("##### Kinematic Tree #####")
        for i in range(1, self.__rmodel.njoints):
            joint_name = self.__rmodel.names[i]
            joint_id = self.__rmodel.getJointId(joint_name)
            parent_joint_id = self.__rmodel.parents[joint_id]
            parent_joint_name = self.__rmodel.names[parent_joint_id]
            print(f"Joint:{joint_name}: id=[{joint_id}], Parent joint: {parent_joint_name}: id=[{parent_joint_id}]")
            print(self.__rmodel.inertias[i], "\n")
    
    def __init_motion_subspace_dict(self):
        # Creat a dictionary of the motion subcapce matrices of all the joints
        self.motion_subcpace = dict()
        for i in range(1, self.__rmodel.njoints):
            joint = self.__rmodel.joints[i]
            joint_type = joint.shortname()
            if joint_type == "JointModelFreeFlyer":
                self.motion_subcpace[i] = np.eye(6)
            elif joint_type == "JointModelRX":
                self.motion_subcpace[i] = np.array([0, 0, 0, 1, 0, 0])
            elif joint_type == "JointModelRY":
                self.motion_subcpace[i] = np.array([0, 0, 0, 0, 1, 0])
            elif joint_type == "JointModelRZ":
                self.motion_subcpace[i] = np.array([0, 0, 0, 0, 0, 1])
            # TODO: Add other joint types if needed, e.g. prismatic
    
    def __cross_operator(self, vec):
        # This is equal to pin.skew(vec)
        return np.array([[ 0     , -vec[2],  vec[1]],
                         [ vec[2],  0     , -vec[0]],
                         [-vec[1],  vec[0],  0     ]])
    
    def __braket_operator(self, vec):
        return np.array([[vec[0], vec[1], vec[2], 0     , 0     , 0     ],
                         [0     , vec[0], 0     , vec[1], vec[2], 0     ],
                         [0     , 0     , vec[0], 0     , vec[1], vec[2]]])
    
    def __update_fk(self, q, dq, ddq):
        pin.forwardKinematics(self.__rmodel, self.__rdata, q, dq, ddq)
        pin.framesForwardKinematics(self.__rmodel, self.__rdata, q)
        pin.computeJointJacobians(self.__rmodel, self.__rdata, q)
    
    def __compute_J_c(self, q, contact_scedule):
        # Returns Jacobian of m feet in contact, dim(3*m,18)
        self.__update_fk(q)
        m = np.sum(contact_scedule)
        J_c = np.zeros((3 * m, self.nv))
        j = 0
        for index in range(self.__nb_ee):
            if contact_scedule[index]:
                frame_id = self.__endeff_ids[index]
                J_c[0+j:3+j, :] = pin.getFrameJacobian(self.__rmodel, self.__rdata, frame_id, pin.LOCAL_WORLD_ALIGNED)[0:3, :]
                j += 3
        return J_c
    
    def __compute_null_space_proj(self, q):
        # Returns null space projector, dim(18, 18)
        J_c = self.__compute_J_c(q)
        p = np.eye((self.nv)) - pinv(J_c) @ J_c
        return p
    
    def __compute_spatial_vel_acc(self):
        # Returns dictionaries of spatial velocity and acceleration of all joints
        spatial_velocities = dict()
        spatial_accelerations = dict()
        
        # Loop over the joint frames
        for i in range(1, self.__rmodel.njoints):
            joint_name = self.__rmodel.names[i]
            joint_id = self.__rmodel.getJointId(joint_name)
            
            # We use the operational frames attached to the joint frames to get v and a expressed in local_world_aligned
            frame_id = self.__rmodel.getFrameId(joint_name)
            joint_spatial_v = pin.getFrameVelocity(
                self.__rmodel,
                self.__rdata,
                frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            joint_spatial_a = pin.getFrameAcceleration(
                self.__rmodel,
                self.__rdata,
                frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            spatial_velocities[joint_id] = joint_spatial_v
            spatial_accelerations[joint_id] = joint_spatial_a
        return spatial_velocities, spatial_accelerations
    
    def __compute_individual_regressor(self, v, a):
        # Returns the regressor matrix, (dim:6x10), for an individual link
        # v, a are spatial velocity and acceleration
        lin_vel = v.linear
        omega = v.angular
        lin_acc = a.linear
        alpha = a.angular
        Y = np.zeros((6, 10), dtype=np.float32)
        Y[0:3, 0] = lin_acc
        Y[0:3, 1:4] =  self.__cross_operator(alpha) + self.__cross_operator(omega) @ self.__cross_operator(omega)
        Y[3:6, 1:4] = -self.__cross_operator(lin_acc)
        Y[3:6, 4:10] = self.__braket_operator(alpha) + self.__cross_operator(omega) @ self.__braket_operator(omega) 
        return Y
    
    def compute_regressor_matrix(self, q, dq, ddq):
        # Returns the global regressor matrix
        # Forward kinematics
        self.__update_fk(q, dq, ddq)
        
        # Compute the indivdual regressors
        ind_regressors = dict()
        spatial_velocities, spatial_accelerations = self.__compute_spatial_vel_acc()
        for joint_id in range(1, self.__rmodel.njoints):
            Y_ind = self.__compute_individual_regressor(spatial_velocities[joint_id], spatial_accelerations[joint_id])
            ind_regressors[joint_id] = Y_ind
        
        # Place individual regressors into the global regressor matrix
        for joint_id in reversed(range(1, self.__rmodel.njoints)):
            # Compute the corrsponding indecies for the columns
            col_start = 10 * (joint_id - 1)
            col_end = col_start + 10
            print("#####", joint_id)
            if self.__floating_base:
                if joint_id == 1:
                    # For the floating base, we place the regressor of the base in the first 6 rows
                    self.__Y[0:6, col_start:col_end] = ind_regressors[joint_id]
                else:
                    # For other joints, reduce the regressor matrix size (have to check: jacobian = pin.getJointJacobian(self.rmodel, self.__rdata, i, pin.LOCAL)[:, 6+(i-2)])
                    joint_regressor = self.motion_subcpace[joint_id].T @ ind_regressors[joint_id]
                    # Place it in the corresponding row
                    row_index = 6+(joint_id-2) # revolutes joint's ids start from 2
                    self.__Y[row_index, col_start:col_end] = joint_regressor
            else:
                # For fixed base, place the projected regressor in the corresponding row
                row_index = joint_id-1 # for fixed base, revolutes joint's ids start from 1 
                self.__Y[row_index, col_start:col_end] = self.motion_subcpace[joint_id].T @ ind_regressors[joint_id]
            
            # Propagate child regresssor back to the parents
            parent_id = self.__rmodel.parents[joint_id]
            jXi = self.__rdata.oMi[parent_id].inverse() * self.__rdata.oMi[joint_id] # Transformation from child to parent frame
            Y_i = jXi.action @ ind_regressors[joint_id] # Transform the regressor
            while parent_id != 0:
                print(joint_id, "----", parent_id)
                # Add the projected regressor to the corresponding row
                row_index = parent_id-1
                self.__Y[row_index, col_start:col_end] = self.motion_subcpace[parent_id].T @ Y_i
                
                # Update parent information
                parent_id = self.__rmodel.parents[parent_id]
                jXi = self.__rdata.oMi[parent_id].inverse() * self.__rdata.oMi[joint_id]
                Y_i = jXi.action @ ind_regressors[joint_id] # TODO: or Y_i
        
        return self.__Y
    
    def get_robot_model(self):
        return self.__rmodel
    
    def get_regressor_pin(self, q, dq, ddq):
        # For validation with pinocchio
        y = pin.computeJointTorqueRegressor(self.__rmodel, self.__rdata, q, dq, ddq)
        tau = pin.rnea(self.__rmodel, self.__rdata, q, dq, ddq)
        return y, tau


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"urdf"/"2dof_plannar_robot.urdf"    
    robot_sys_iden = SysIdentification(str(robot_urdf), floating_base=False)
    
    robot_q = pin.randomConfiguration(robot_sys_iden.get_robot_model())
    robot_dq = np.random.rand(robot_sys_iden.nv)
    robot_ddq = np.random.rand(robot_sys_iden.nv)
    
    regressor = robot_sys_iden.compute_regressor_matrix(robot_q, robot_dq, robot_ddq)
    print("#### Computed Regressor ####\n", regressor)
    
    y, tau = robot_sys_iden.get_regressor_pin(robot_q, robot_dq, robot_ddq)
    print("#### Pinocchio #### \n",y)
