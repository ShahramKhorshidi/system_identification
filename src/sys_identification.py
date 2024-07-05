import yaml
import trimesh
import numpy as np
import pinocchio as pin
from pathlib import Path
from numpy.linalg import pinv
from urdf_parser_py.urdf import URDF, Box, Cylinder, Sphere, Mesh


class SystemIdentification(object):
    def __init__(self, urdf_filename, floating_base, viz=None):
        self.urdf_path = urdf_filename
        # Creat robot model and data
        self._floating_base = floating_base
        if self._floating_base:
            self._rmodel = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        else:
            self._rmodel = pin.buildModelFromUrdf(self.urdf_path)
        self._rdata = self._rmodel.createData()
        
        # Set the gravity vector in pinocchio
        self._rmodel.gravity.linear = np.array([0, 0, -9.81])
        
        # Dimensions of robot confgiuration space and velocity vector
        self.nq = self._rmodel.nq
        self.nv = self._rmodel.nv
        
        # Selection matrix
        if floating_base:
            self._S = np.zeros((self.nv-6, self.nv))
            self._S[:, 6:] = np.eye(self.nv-6)
        else:
            self._S = np.eye(self.nv)
        
        # Initialize the regressor matrix with proper dimension
        # For now only considering 10 inertial parameters for each link
        # [m h_x h_y h_z I_xx I_xy I_xz I_yy I_yz I_zz]
        self._num_inertial_param = 10
        self._num_links = self._rmodel.njoints-1 # In pinocchio, universe is always in the kinematic tree with joint[id]=0
        self._Y = np.zeros((self.nv, self._num_inertial_param * self._num_links), dtype=np.float32)
        
        # List of the end_effector names
        # TODO: Later put all changing parameters in a separate yaml config file
        self._end_eff_frame_names = ["HL_ANKLE", "HR_ANKLE", "FL_ANKLE", "FR_ANKLE"]
        self._endeff_ids = [
            self._rmodel.getFrameId(name)
            for name in self._end_eff_frame_names
        ]
        self._nb_ee = len(self._end_eff_frame_names)
        
        self._init_motion_subspace_dict()
        self._show_kinematic_tree()
        self.get_bounding_ellipsoids()
    
    def _show_kinematic_tree(self):
        print("##### Kinematic Tree #####")
        for i in range(1, self._rmodel.njoints):
            joint_name = self._rmodel.names[i]
            joint_id = self._rmodel.getJointId(joint_name)
            joint_type = self._rmodel.joints[i].shortname()
            parent_joint_id = self._rmodel.parents[joint_id]
            parent_joint_name = self._rmodel.names[parent_joint_id]
            print(f"Name:{joint_name},id=[{joint_id}],type:{joint_type} -- Parent:{parent_joint_name},id=[{parent_joint_id}]")
            print(self._rmodel.inertias[i], "\n")
    
    def _init_motion_subspace_dict(self):
        # Creat a dictionary of the motion subcapce matrices of all the joints
        self._motion_subcpace = dict()
        for i in range(1, self._rmodel.njoints):
            joint = self._rmodel.joints[i]
            joint_type = joint.shortname()
            if joint_type == "JointModelFreeFlyer":
                self._motion_subcpace[i] = np.eye(6)
            elif joint_type == "JointModelRX":
                self._motion_subcpace[i] = np.array([0, 0, 0, 1, 0, 0])
            elif joint_type == "JointModelRY":
                self._motion_subcpace[i] = np.array([0, 0, 0, 0, 1, 0])
            elif joint_type == "JointModelRZ":
                self._motion_subcpace[i] = np.array([0, 0, 0, 0, 0, 1])
            # TODO: Add other joint types if needed, e.g. prismatic
    
    def _cross_operator(self, vec):
        # This is equal to pin.skew(vec)
        return np.array([[ 0     , -vec[2],  vec[1]],
                         [ vec[2],  0     , -vec[0]],
                         [-vec[1],  vec[0],  0     ]])
    
    def _braket_operator(self, vec):
        return np.array([[vec[0], vec[1], vec[2], 0     , 0     , 0     ],
                         [0     , vec[0], 0     , vec[1], vec[2], 0     ],
                         [0     , 0     , vec[0], 0     , vec[1], vec[2]]])
    
    def _update_fk(self, q, dq, ddq):
        # Update the forward kinematics of the robot
        pin.forwardKinematics(self._rmodel, self._rdata, q, dq, ddq)
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)
        pin.computeJointJacobians(self._rmodel, self._rdata, q)
    
    def _compute_J_c(self, contact_scedule):
        # Returns the jacobian of m feet in contact, dim(3*m,18)
        m = int(np.sum(contact_scedule))
        J_c = np.zeros((3 * m, self.nv))
        j = 0
        for index in range(self._nb_ee):
            if contact_scedule[index]:
                frame_id = self._endeff_ids[index]
                J_c[0+j:3+j, :] = pin.getFrameJacobian(self._rmodel, self._rdata, frame_id, pin.LOCAL_WORLD_ALIGNED)[0:3, :]
                j += 3
        return J_c
    
    def _compute_null_space_proj(self, contact_scedule):
        # Returns null space projector, dim(18, 18)
        J_c = self._compute_J_c(contact_scedule)
        p = np.eye((self.nv)) - pinv(J_c) @ J_c
        return p
    
    def _compute_spatial_vel_acc(self):
        # Returns dictionaries of spatial velocity and acceleration of all joints
        spatial_velocities = dict()
        spatial_accelerations = dict()
        
        # Loop over the joint frames
        for i in range(1, self._rmodel.njoints):
            joint_name = self._rmodel.names[i]
            joint_id = self._rmodel.getJointId(joint_name)
            
            # We use the operational frames attached to the joint frames to get v and a expressed in local_world_aligned
            frame_id = self._rmodel.getFrameId(joint_name)
            joint_spatial_v = pin.getFrameVelocity(
                self._rmodel,
                self._rdata,
                frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            joint_spatial_a = pin.getFrameAcceleration(
                self._rmodel,
                self._rdata,
                frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            spatial_velocities[joint_id] = joint_spatial_v
            spatial_accelerations[joint_id] = joint_spatial_a
        return spatial_velocities, spatial_accelerations
    
    def _compute_individual_regressor(self, v, a):
        # Returns the regressor matrix, (dim:6x10), for an individual link
        # v, a are spatial velocity and acceleration
        lin_vel = v.linear
        omega = v.angular
        lin_acc = a.linear
        alpha = a.angular
        Y = np.zeros((6, 10), dtype=np.float32)
        Y[0:3, 0] = lin_acc
        Y[0:3, 1:4] =  self._cross_operator(alpha) + self._cross_operator(omega) @ self._cross_operator(omega)
        Y[3:6, 1:4] = -self._cross_operator(lin_acc)
        Y[3:6, 4:10] = self._braket_operator(alpha) + self._cross_operator(omega) @ self._braket_operator(omega) 
        return Y
    
    def _compute_regressor_matrix(self):
        """
        Returns the global regressor matrix
        """
        # Compute the indivdual regressors
        ind_regressors = dict()
        spatial_velocities, spatial_accelerations = self._compute_spatial_vel_acc()
        for joint_id in range(1, self._rmodel.njoints):
            Y_ind = self._compute_individual_regressor(spatial_velocities[joint_id], spatial_accelerations[joint_id])
            ind_regressors[joint_id] = Y_ind
        
        # Place individual regressors into the global regressor matrix
        for joint_id in reversed(range(1, self._rmodel.njoints)):
            # Compute the corrsponding indecies for the columns
            col_start = 10 * (joint_id - 1)
            col_end = col_start + 10
            if self._floating_base:
                if joint_id == 1:
                    # For the floating base, we place the regressor of the base in the first 6 rows
                    self._Y[0:6, col_start:col_end] = ind_regressors[joint_id]
                else:
                    # For other joints, reduce the regressor matrix size (have to check: jacobian = pin.getJointJacobian(self.rmodel, self.__rdata, i, pin.LOCAL)[:, 6+(i-2)])
                    joint_regressor = self._motion_subcpace[joint_id].T @ ind_regressors[joint_id]
                    # Place it in the corresponding row
                    row_index = 6+(joint_id-2) # revolutes joint's ids start from 2
                    self._Y[row_index, col_start:col_end] = joint_regressor
            else:
                # For fixed base, place the projected regressor in the corresponding row
                row_index = joint_id-1 # for fixed base, revolutes joint's ids start from 1 
                self._Y[row_index, col_start:col_end] = self._motion_subcpace[joint_id].T @ ind_regressors[joint_id]
            
            # Propagate child regresssor back to the parents
            parent_id = self._rmodel.parents[joint_id]
            jXi = self._rdata.oMi[parent_id].inverse() * self._rdata.oMi[joint_id] # Transformation from child to parent frame
            Y_i = jXi.action @ ind_regressors[joint_id] # Transform the regressor
            while parent_id != 0:
                # Add the projected regressor to the corresponding row
                row_index = parent_id-1
                self._Y[row_index, col_start:col_end] = self._motion_subcpace[parent_id].T @ Y_i
                
                # Update parent information
                parent_id = self._rmodel.parents[parent_id]
                jXi = self._rdata.oMi[parent_id].inverse() * self._rdata.oMi[joint_id]
                Y_i = jXi.action @ ind_regressors[joint_id]
        
        return self._Y
    
    def get_bounding_ellipsoids(self):
        robot = URDF.from_xml_file(self.urdf_path)
        bounding_ellipsoids = []
        for link in robot.links:
            print(link)
            for visual in link.visuals:
                geometry = visual.geometry
                if isinstance(geometry, Box):
                    size = np.array(geometry.size)
                    semi_axes = size / 2
                    center = visual.origin.xyz if visual.origin else [0, 0, 0]
                elif isinstance(geometry, Cylinder):
                    radius = geometry.radius
                    length = geometry.length
                    semi_axes = [radius, radius, length / 2]
                    center = visual.origin.xyz if visual.origin else [0, 0, 0]
                elif isinstance(geometry, Sphere):
                    radius = geometry.radius
                    semi_axes = [radius, radius, radius]
                    center = visual.origin.xyz if visual.origin else [0, 0, 0]
                elif isinstance(geometry, Mesh):
                    mesh_path = geometry.filename
                    mesh = trimesh.load_mesh(mesh_path)
                    # TODO: Check if this is actually returning half of the lengths for the bounding_box 
                    # otherwise it should be divided by 2
                    semi_axes = mesh.bounding_box.extents
                    center = mesh.bounding_box.centroid
                else:
                    raise ValueError(f"Unsupported geometry type for link {link.name}")
                
                bounding_ellipsoids.append({'semi_axes': semi_axes, 'center': center})
        print(bounding_ellipsoids)
        print("#####", len(bounding_ellipsoids))
        return bounding_ellipsoids
    
    def get_projected_llsq_roblem(self, q, dq, ddq, tau, contact_scedule):
        # Returns the regressor matrix and joint torque vector projected into the null space of contacts
        self._update_fk(q, dq, ddq)
        Y = self._compute_regressor_matrix()
        P = self._compute_null_space_proj(contact_scedule)
        Y_proj = P @ Y
        tau_proj = P @ self._S.T @ tau
        return Y_proj, tau_proj
    
    def get_regressor_pin(self, q, dq, ddq, tau, contact_scedule):
        # For validation with pinocchio
        self._update_fk(q, dq, ddq)
        Y = pin.computeJointTorqueRegressor(self._rmodel, self._rdata, q, dq, ddq)
        P = self._compute_null_space_proj(contact_scedule)
        Y_proj = P @ Y
        tau_proj = P @ self._S.T @ tau
        return Y_proj, tau_proj


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"urdf"/"robot.urdf"
    robot_sys_iden = SystemIdentification(str(robot_urdf), floating_base=True)
    
    # robot_q = pin.randomConfiguration(robot_sys_iden._rmodel)
    robot_q = np.random.rand(robot_sys_iden.nq)
    robot_dq = np.random.rand(robot_sys_iden.nv)
    robot_ddq = np.random.rand(robot_sys_iden.nv)
    robot_tau = np.random.rand(robot_sys_iden.nv-6)
    contact_config  = np.array([1, 1, 1, 1])
    
    # regressor = robot_sys_iden.compute_regressor_matrix(robot_q, robot_dq, robot_ddq)
    # print("#### Computed Regressor ####\n", regressor)
    y, tau = robot_sys_iden.get_regressor_pin(robot_q, robot_dq, robot_ddq, robot_tau, contact_config)
    # print("#### Pinocchio #### \n",y)
