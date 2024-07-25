import yaml
import trimesh
import numpy as np
import pinocchio as pin
from numpy.linalg import pinv
from urdf_parser_py.urdf import URDF, Box, Cylinder, Sphere, Mesh


class SystemIdentification(object):
    def __init__(self, urdf_file, config_file, floating_base):
        self._urdf_path = urdf_file
        
        # Create robot model and data
        self._floating_base = floating_base
        if self._floating_base:
            self._rmodel = pin.buildModelFromUrdf(self._urdf_path, pin.JointModelFreeFlyer())
        else:
            self._rmodel = pin.buildModelFromUrdf(self._urdf_path)
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
        # inertial parameters for each link, phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]
        self._num_inertial_params = 10
        self._num_links = self._rmodel.njoints-1 # In pinocchio, universe is always in the kinematic tree with joint[id]=0
        self._phi_prior = np.zeros((self._num_inertial_params * self._num_links), dtype=np.float32)
        self._Y = np.zeros((self.nv, self._num_inertial_params * self._num_links), dtype=np.float32)
        
        # Load robot configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        robot_config = config.get('robot', {})
        self._robot_name = robot_config.get('name')
        self._robot_mass = robot_config.get('mass')
        
        # List of bounding ellipsoids for all links
        self._bounding_ellipsoids = []
        for ellipsoid in robot_config.get('bounding_ellipsoids'):
            ellipsoid['semi_axes'] = np.array(ellipsoid['semi_axes'])
            ellipsoid['center'] = np.array(ellipsoid['center'])
            self._bounding_ellipsoids.append(ellipsoid)
        
        # List of the end_effector names
        self._end_eff_frame_names = robot_config.get('end_effectors_frame_names', [])
        self._endeff_ids = [
            self._rmodel.getFrameId(name)
            for name in self._end_eff_frame_names
        ]
        self._nb_ee = len(self._end_eff_frame_names)
        
        self._init_motion_subspace_dict()
        # self._show_kinematic_tree()
    
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
    
    def _compute_lambda(self, force, contact_scedule):
        # Returns the force vector of m feet in contact, dim(3*m)
        m = int(np.sum(contact_scedule))
        lamda = np.zeros(3 * m)
        j = 0
        for i in range(self._nb_ee):
            if contact_scedule[i]:
                lamda[0+j:3+j] = force[3*i:3*(i+1)]
                j += 3
        return lamda
    
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
        omega = v.angular
        lin_acc = a.linear - np.array([0, 0, -9.81])
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

    def compute_bounding_ellipsoids(self):
        robot = URDF.from_xml_file(self._urdf_path)
        self._bounding_ellipsoids = []
        for link in robot.links:
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
                    semi_axes = mesh.bounding_box.extents / 2
                    origin =  visual.origin.xyz
                    center =  mesh.bounding_box.centroid + origin
                else:
                    raise ValueError(f"Unsupported geometry type for link {link.name}")
                self._bounding_ellipsoids.append({'semi_axes': semi_axes, 'center': center})

    def get_robot_mass(self):
        return self._robot_mass
    
    def get_num_links(self):
        return self._num_links
    
    def get_bounding_ellipsoids(self):
        return self._bounding_ellipsoids
    
    def get_phi_prior(self):
        for i in range(1, self._rmodel.njoints):
            j = 10*(i-1)
            self._phi_prior[j] = self._rmodel.inertias[i].mass
            self._phi_prior[j+1: j+4] = self._rmodel.inertias[i].mass * self._rmodel.inertias[i].lever
            self._phi_prior[j+4: j+7] = self._rmodel.inertias[i].inertia[0, :]
            self._phi_prior[j+7: j+9] = self._rmodel.inertias[i].inertia[1, 1:]
            self._phi_prior[j+9] = self._rmodel.inertias[i].inertia[2, 2]
        return self._phi_prior
    
    def check_physical_consistency(self, phi):
        # Returns the minimum eigenvalue of matrices in LMI constraints
        # For phiysical consistency all values should be non-negative
        eigval_I_bar = []
        eigval_I = [] # Spatial body inertia
        eigval_J = [] # Pseudo inertia
        eigval_com =[]
        trace_JQ = []
        
        if len(self._bounding_ellipsoids)==0:
            self.compute_bounding_ellipsoids()
        
        for j in range(0, phi.size, self._num_inertial_params):
            # Extracting the inertial parameters
            m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi[j: j+self._num_inertial_params]
            h = np.array([h_x, h_y, h_z])
            ellipsoid_params = self._bounding_ellipsoids[j // self._num_inertial_params]
            semi_axes = ellipsoid_params['semi_axes']
            center = ellipsoid_params['center']
            
            # Inertia matrix (3x3)
            I_bar = np.array([[I_xx, I_xy, I_xz],
                              [I_xy, I_yy, I_yz],
                              [I_xz, I_yz, I_zz]])
            
            # Spatial body inertia (6x6)
            I = np.zeros((6,6), dtype=np.float32)
            I[0:3, 0:3] = I_bar
            I[0:3, 3:] = pin.skew(h)
            I[3:, 0:3] = pin.skew(h).T
            I[3:, 3:] = m * np.eye(3)
            
            # Bounding ellipsoid: Q matrix (4x4)
            Q_full = np.zeros((4,4), dtype=np.float32)
            Q = np.linalg.inv(np.diag(semi_axes)**2)
            Q_full[:3, :3] = Q
            Q_full[:3, 3] = Q @ center
            Q_full[3, :3] = (Q @ center).T
            Q_full[3, 3] = 1 - (center.T @ Q @ center)
            
            # Pseudo inertia matrix (4x4)
            J = np.zeros((4,4), dtype=np.float32)
            J[:3, :3] = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
            J[:3, 3] = h
            J[3, :3] = h.T 
            J[3,3] = m
            
            # CoM constraint (4x4)
            C = np.zeros((4,4), dtype=np.float32)
            C[0, 0] = m
            C[0, 1:] = h.T - m * center.T
            C[1:, 0] = h - m * center
            C[1:, 1:] = m * np.diag(semi_axes)**2
            
            # Calculate minimum eigenvalue of each matrix
            min_eigval_I_bar = np.min(np.linalg.eigvals(I_bar))
            min_eigval_I = np.min(np.linalg.eigvals(I))
            min_eigval_J = np.min(np.linalg.eigvals(J))
            min_eigval_c = np.min(np.linalg.eigvals(C))
            density_realizable = np.trace(J @ Q_full)
            
            # Add to the list
            eigval_I_bar.append(min_eigval_I_bar)
            eigval_I.append(min_eigval_I)
            eigval_J.append(min_eigval_J)
            eigval_com.append(min_eigval_c)
            trace_JQ.append(density_realizable)
        return eigval_I_bar, eigval_I, eigval_J, eigval_com, trace_JQ
    
    def get_projected_llsq_problem(self, q, dq, ddq, tau, contact_scedule):
        # Returns the regressor matrix and joint torque vector projected into the null space of contacts
        self._update_fk(q, dq, ddq)
        Y = self._compute_regressor_matrix()
        P = self._compute_null_space_proj(contact_scedule)
        Y_proj = P @ Y
        tau_proj = P @ self._S.T @ tau
        return Y_proj, tau_proj
    
    def get_full_regressor_force(self, q, dq, ddq, tau, ee_force, cnt):
        # Returns regressor matrix and force vector: (Y @ phi = force)
        # for floating base dynamics: M(q) @ ddq + h(q, dq) = S^T @ tau + J_c^T @ lambda
        self._update_fk(q, dq, ddq)
        Y = pin.computeJointTorqueRegressor(self._rmodel, self._rdata, q, dq, ddq)
        J_c = self._compute_J_c(cnt)
        lamda = self._compute_lambda(ee_force, cnt)
        F = self._S.T @ tau + J_c.T @ lamda
        return Y, F 
    
    def get_proj_regressor_torque(self, q, dq, ddq, tau, cnt):
        # Returns regressor matrix and torque vector projected into 
        # the null-space of contatc Jacobian: (P @ Y) @ phi = (P @ tau)
        # for motion dynamics: P @ (M(q) @ ddq + h(q, dq)) = P @ S^T @ tau
        self._update_fk(q, dq, ddq)
        Y = pin.computeJointTorqueRegressor(self._rmodel, self._rdata, q, dq, ddq)
        P = self._compute_null_space_proj(cnt)
        Y_proj = P @ Y
        tau_proj = P @ self._S.T @ tau
        return Y_proj, tau_proj
