import yaml
import trimesh
import numpy as np
import pinocchio as pin
from numpy.linalg import pinv
from abc import ABC, abstractmethod
from urdf_parser_py.urdf import URDF, Box, Cylinder, Sphere, Mesh


class RigidBodyDynamics(ABC):
    def __init__(
            self,
            urdf_file: str,
            config_file: str,
            mesh_dir: str,
            floating_base: bool,
            info: bool = False
        ):
        self.urdf_path = urdf_file
        # Create robot model and data
        self.floating_base = floating_base
        if self.floating_base:
            self.rmodel = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        else:
            self.rmodel = pin.buildModelFromUrdf(self.urdf_path)
        self.rdata = self.rmodel.createData()
        
        self.g = self.rmodel.gravity.linear
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        
        # Selection matrix
        if self.floating_base:
            self.base_dof = 6
            self.joints_dof = self.nv - self.base_dof
            self.S = np.zeros((self.joints_dof, self.nv))
            self.S[:, self.base_dof:] = np.eye(self.joints_dof)
        else:
            self.base_dof = 0
            self.joints_dof = self.nv
            self.S = np.eye(self.joints_dof)
        
        # Load robot configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        robot_config = config.get('robot', {})
        self.robot_name = robot_config.get('name')
        self.robot_mass = robot_config.get('mass')

        # List of link names
        self.link_names = robot_config.get('link_names', [])
        
        # List of end_effector names
        self.end_eff_frame_names = robot_config.get('end_effectors_frame_names', [])
        self.endeff_ids = [
            self.rmodel.getFrameId(name)
            for name in self.end_eff_frame_names
        ]
        self.nb_ee = len(self.end_eff_frame_names)
        
        # Initialize the regressor matrix with proper dimension
        # Inertial parameters for each link with Pinocchio order:
        # phi = [m, h_x, h_y, h_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        self.phi_dim = 10
        self.num_links = self.rmodel.njoints - 1
        self.Y = np.zeros((self.nv, self.phi_dim * self.num_links), dtype=np.float64)

        # Initialize the friction regressors
        self.B_v = np.eye(self.joints_dof) # Viscous friction
        self.B_c = np.eye(self.joints_dof) # Coulomb friction
        
        # Pinocchio uses [f; n] for spatial force, but here we use [n; f]
        # Maps [n; f] <-> [f; n]
        self._force_swap = np.block([
            [np.zeros((3,3)), np.eye(3)],
            [np.eye(3),       np.zeros((3,3))]
        ])

        # Initialization
        self.bounding_ellipsoids = None
        if mesh_dir is not None:
            self.mesh_dir = mesh_dir
            self._compute_bounding_ellipsoids()
        self._init_motion_subspace_dict()
        self._compute_phi_nom_pin()
        self._init_com_frames()
        if info:
            self._show_kinematic_tree()

    # -------------------- Internal methods -------------------- #    
    def _show_kinematic_tree(self):
        print("-"*20," Kinematic Tree ","-"*20)
        for i in range(1, self.rmodel.njoints):
            joint_name = self.rmodel.names[i]
            joint_id = self.rmodel.getJointId(joint_name)
            joint_type = self.rmodel.joints[i].shortname()
            parent_joint_id = self.rmodel.parents[joint_id]
            parent_joint_name = self.rmodel.names[parent_joint_id]
            print(f"Name:{joint_name},\id=[{joint_id}],type:{joint_type} -- Parent:{parent_joint_name},id=[{parent_joint_id}]")
            print(self.rmodel.inertias[i], "\n") # mass, lever, inertia (around CoM)
            
    def _init_motion_subspace_dict(self):
        # Create a dictionary of the motion subspace matrices of all the joints.
        # Featherstone/Pinocchio store angular on top, linear on bottom for motion vectors [w; v]
        self.motion_subspace = dict()
        for i in range(1, self.rmodel.njoints):
            joint = self.rmodel.joints[i]
            joint_type = joint.shortname()
            if joint_type == "JointModelFreeFlyer":
                # In Pinocchio the free-flyer joint’s generalized velocity is ordered [v; w]
                self.motion_subspace[i] = np.block([
                    [np.zeros((3,3)), np.eye(3)],
                    [np.eye(3),       np.zeros((3,3))]
                ])
            elif joint_type == "JointModelRX":
                self.motion_subspace[i] = np.array([1, 0, 0, 0, 0, 0])
            elif joint_type == "JointModelRY":
                self.motion_subspace[i] = np.array([0, 1, 0, 0, 0, 0])
            elif joint_type == "JointModelRZ":
                self.motion_subspace[i] = np.array([0, 0, 1, 0, 0, 0])
            elif joint_type == "JointModelPX":
                self.motion_subspace[i] = np.array([0, 0, 0, 1, 0, 0])
            elif joint_type == "JointModelPY":
                self.motion_subspace[i] = np.array([0, 0, 0, 0, 1, 0])
            elif joint_type == "JointModelPZ":
                self.motion_subspace[i] = np.array([0, 0, 0, 0, 0, 1])
            else:
                # Extend here for other joint types (spherical, planar, etc.)
                raise NotImplementedError(f"Joint type {joint_type} not handled in S_i.")

    def _init_com_frames(self):
        """
        For each joint j, create a Pinocchio frame located at the link's COM
        (given by model.inertias[j].lever) and store its frame id.
        """
        self.com_frame_ids = []

        for j in range(1, self.rmodel.njoints):
            joint_name = self.rmodel.names[j]
            inertia = self.rmodel.inertias[j]
            com_offset = inertia.lever.copy()  # 3D vector in joint frame

            # Placement of COM relative to joint frame: rotation=I, translation=com_offset
            placement = pin.SE3(np.eye(3), com_offset)

            # We attach the frame to joint j, and choose the joint's frame as parent frame.
            # parentFrame = frame of the joint itself:
            parent_frame_id = self.rmodel.getFrameId(joint_name)

            frame_name = joint_name + "_COM"
            frame = pin.Frame(
                frame_name,
                j,                # parentJoint
                parent_frame_id,  # parentFrame
                placement,
                pin.BODY
            )
            fid = self.rmodel.addFrame(frame)
            self.com_frame_ids.append(fid)

    def _compute_phi_nom_pin(self):
        # Compute the nominal inertial parameters using Pinocchio's internal representation
        self.phi_nom = np.zeros(self.phi_dim * self.num_links, dtype=np.float64)
        for jid in range(1, self.rmodel.njoints):
            self.phi_nom[10*(jid-1):10*jid] = self.rmodel.inertias[jid].toDynamicParameters()

    def _compute_inertial_params(self):
        # Compute the inertial parameters for each link from URDF
        # The CoM is expressed in the body farme about the joint origin
        # The inertia matrix is expressed about the CoM frame
        self.inertial_params = []
        robot = URDF.from_xml_file(self.urdf_path)
        for link in robot.links:
            if link.name in self.link_names:
                m = link.inertial.mass
                com = np.array(link.inertial.origin.xyz)
                rpy = np.array(link.inertial.origin.rpy)
                I_xx = link.inertial.inertia.ixx
                I_xy = link.inertial.inertia.ixy
                I_xz = link.inertial.inertia.ixz
                I_yy = link.inertial.inertia.iyy
                I_yz = link.inertial.inertia.iyz
                I_zz = link.inertial.inertia.izz
                I_c = np.array([[I_xx, I_xy, I_xz],
                                [I_xy, I_yy, I_yz],
                                [I_xz, I_yz, I_zz]])
                self.inertial_params.append({'mass': m, 'com': com, 'rpy': rpy, 'I_c': I_c})

    def _compute_phi_nom_urdf(self):
        # Compute the nominal inertial parameters vector phi_nom (size=10*num_link); using the URDF file
        # The inertial parameters of each link is expressed w.r.t the body_frame at the joint origin
        num_links = len(self.link_names)
        self.phi_nom = np.zeros((self.phi_dim * num_links), dtype=np.float64)
        self._compute_inertial_params()
        for i in range(num_links):
            inertials = self.inertial_params[i]
            mass  = inertials['mass']
            com = inertials['com'] # CoM position w.r.t the joint frame of the link
            h = mass * com # First moment of inertia
            
            # Inertia matrix
            I_c = inertials['I_c']
            # Order of rotation: roll, pitch, yaw (fixed axes)
            rpy = inertials['rpy']
            R = pin.utils.rpyToMatrix(rpy[0], rpy[1], rpy[2])
            I_rotated = R @ I_c @ R.T
            I_bar = I_rotated + (mass * pin.skew(com) @ pin.skew(com).T)
            Ixx, Iyy, Izz = I_bar[0,0], I_bar[1,1], I_bar[2,2]
            Ixy, Ixz, Iyz = I_bar[0,1], I_bar[0,2], I_bar[1,2]
            # Store the inerta parameters inside phi_prior vector
            j = 10*i
            self.phi_nom[j] = mass
            self.phi_nom[j+1: j+4] = h
            self.phi_nom[j+4: j+10] = Ixx, Ixy, Iyy, Ixz, Iyz, Izz

    def _compute_bounding_ellipsoids(self):
        self.bounding_ellipsoids = []
        robot = URDF.from_xml_file(self.urdf_path)
        for link in robot.links:
            if link.name in self.link_names:
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
                        mesh_path = self.mesh_dir + "/" +  link.name + ".obj"
                        mesh = trimesh.load_mesh(mesh_path)
                        semi_axes = mesh.bounding_box.extents / 2
                        origin =  visual.origin.xyz
                        center =  mesh.bounding_box.centroid + origin
                    else:
                        raise ValueError(f"Unsupported geometry type for link {link.name}")
                    self.bounding_ellipsoids.append({'semi_axes': semi_axes, 'center': center})

    @abstractmethod
    def _contact_block_dim(self) -> int:
        """3 for point contacts, 6 for full wrenches."""
    
    @abstractmethod
    def _active_contact_frames(self, cnt_schedule: np.ndarray) -> list[int]:
        """Return frame ids of active contacts in a consistent order."""

    @abstractmethod
    def _compute_J_c(self, cnt_schedule: np.ndarray) -> np.ndarray:
        """Stacked contact Jacobian with shape (m*k, nv), k = _contact_block_dim()."""

    @abstractmethod
    def _compute_lambda_c(self, ee_wrenches: np.ndarray, cnt_schedule: np.ndarray) -> np.ndarray:
        """Stacked lambda (m*k,), consistent with _compute_J_c ordering & block size."""
    
    def _compute_null_space_proj(self, contact_scedule):
        # Returns null space projector, dim(18, 18)
        J_c = self._compute_J_c(contact_scedule)
        p = np.eye((self.nv)) - pinv(J_c) @ J_c
        return p
    
    def _braket_operator(self, vec: np.ndarray) -> np.ndarray:
        # I_bar.(omega) = bracekt(omega).[Ixx, Ixy, Iyy, Ixz, Iyz, Izz].T
        x, y, z = vec
        return np.array([
            [x, y, 0, z, 0, 0],
            [0, x, y, 0, z, 0],
            [0, 0, 0, x, y, z]
        ], dtype=np.float64)
    
    def _compute_spatial_vel_acc(self) -> tuple:
        # Returns dictionaries of spatial velocity and acceleration of all joints
        spatial_vel = dict()
        spatial_acc = dict()
        spatial_vel = {i: self.rdata.v[i] for i in range(1, self.rmodel.njoints)}
        spatial_acc = {i: self.rdata.a[i] for i in range(1, self.rmodel.njoints)}
        return spatial_vel, spatial_acc

    # -------------------- Classical Regressor -------------------- #
    def _compute_Y_i(self, v_i: pin.Motion, a_i: pin.Motion, R_i: np.ndarray) -> np.ndarray:
        # Returns the regressor matrix, (dim:6x10), for an individual link
        # v, a are spatial velocity and acceleration
        omega = np.asarray(v_i.angular).reshape(3)
        alpha = np.asarray(a_i.angular).reshape(3)
        vlin  = np.asarray(v_i.linear).reshape(3)
        # In Atkenson ddp is the classical joint-origin acceleration
        ddp   = np.asarray(a_i.linear).reshape(3) + np.cross(omega, vlin)
        g_local = R_i.T @ self.g
        a_lin   = ddp - g_local
        
        # Compute the regressor matrix Y_i
        # Row order [n; f] (moment/torque on top, force on bottom)
        Y_i = np.zeros((6, 10), dtype=np.float64)
        Y_i[0:3, 1:4] = pin.skew(-a_lin)
        Y_i[0:3, 4:10] = self._braket_operator(alpha) + pin.skew(omega) @ self._braket_operator(omega)
        Y_i[3:6, 0] = a_lin
        Y_i[3:6, 1:4] = pin.skew(alpha) + pin.skew(omega) @ pin.skew(omega)
        return Y_i

    def _compute_regressor_matrix(self) -> np.ndarray:
        """
        Returns the classical regressor matrix
        Y(q, dq, ddq) @ Phi_all_link = tau_rnea
        """
        # Reset Y for this call
        self.Y.fill(0.0)
        
        # 1) Per-joint spatial kinematics
        spatial_vel, spatial_acc = self._compute_spatial_vel_acc()

        # 2) Per-link local 6x10 regressors Y_i
        ind_regressors = dict()
        for i in range(1, self.rmodel.njoints):
            R_i = self.rdata.oMi[i].rotation.copy()
            ind_regressors[i] = self._compute_Y_i(spatial_vel[i], spatial_acc[i], R_i)
        
        # 3) Place individual regressors into the global regressor matrix
        for i in reversed(range(1, self.rmodel.njoints)):
            # Compute the corresponding indices for the columns
            col_start = 10 * (i - 1)
            col_end = col_start + 10
            S_i = self.motion_subspace[i]

            # If S_i is the motion subspace of a joint, reshape it to a vector
            if S_i.ndim == 1:
                S_i = S_i.reshape(6, 1)
            
            # 3a) Joint own contribution to its own row
            proj_Y_i = S_i.T @ ind_regressors[i]
            
            if self.floating_base and i == 1:
                # Free-flyer: 6 rows at the top
                self.Y[0:6, col_start:col_end] += proj_Y_i
            else:
                # For free-flyer, revolutes joint's ids start from 2
                # For fixed base, revolutes joint's ids start from 1
                row_index = (6 + (i - 2)) if self.floating_base else (i-1)
                self.Y[row_index, col_start:col_end] += proj_Y_i.flatten()

            # 3b) Propagate child regressor up to the tree and project at each parent
            cur = i
            parent_id = self.rmodel.parents[cur]
            Y_up = ind_regressors[i] # this is still expressed in frame 'cur'

            while parent_id != 0:
                # Motion transform from parent to current: ^pX_c
                X_p_c = self.rdata.oMi[parent_id].inverse() * self.rdata.oMi[cur]
                
                # Force action (dual)
                X_p_c_star = X_p_c.toDualActionMatrix()
                
                # Express current block in the parent frame
                Y_up = self._force_swap @ (X_p_c_star @ (self._force_swap @ Y_up))
                
                # Project onto parent's joint subspace row
                S_p = self.motion_subspace[parent_id]
                if S_p.ndim == 1:
                    S_p = S_p.reshape(6, 1)
                proj_Y_p = S_p.T @ Y_up
                
                if parent_id == 1 and self.floating_base:
                    self.Y[0:6, col_start:col_end] += proj_Y_p
                else:
                    row_p = (6 + (parent_id - 2)) if self.floating_base else (parent_id - 1)
                    self.Y[row_p, col_start:col_end] += proj_Y_p.flatten()
                
                # climb one level
                cur = parent_id
                parent_id = self.rmodel.parents[cur]
        return self.Y
    
    def _compute_tau_nom(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        est_link_id: int = None
    ) -> np.ndarray:
        """
        Returns the nominal joint torque vector assuming zero inertials for the estimated link.
        """
        Y_full = pin.computeJointTorqueRegressor(self.rmodel, self.rdata, q, dq, ddq)
        phi_nom = self.phi_nom.copy()
        if est_link_id is not None:
            if est_link_id < 0 or est_link_id >= self.num_links:
                raise ValueError(f"est_link_id {est_link_id} is out of range [0, {self.num_links-1}]")
            i = 10 * est_link_id
            phi_nom[i: i+10] = 0.0
        tau_nom = Y_full @ phi_nom
        return tau_nom
    
    # -------------------- Momentum_based Regressor -------------------- #
    # TODO: Need to add Momentum based regressor, so we avoid ddq measurements
    
    # -------------------- API -------------------- #
    def update_fk(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray):
        if q.shape[0] != self.nq:
            raise ValueError(f"q has {q.shape[0]} elements, expected {self.nq}")
        if dq.shape[0] != self.nv:
            raise ValueError(f"dq has {dq.shape[0]} elements, expected {self.nv}")
        if ddq.shape[0] != self.nv:
            raise ValueError(f"ddq has {ddq.shape[0]} elements, expected {self.nv}")
        pin.forwardKinematics(self.rmodel, self.rdata, q, dq, ddq)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
    
    def build_updated_urdf(self, phi_ident, b_v, b_c):
        # Creat a new URDF file based on identified inertial parameters
        # Load the URDF model
        robot = URDF.from_xml_file(self.urdf_path)

        # Iterate over each link and update the inertial parameters
        idx=0
        for link in robot.links:
            if link.name in self.link_names:
                # Extract the inertial parameters of the link
                mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi_ident[idx: idx+10]
                com = np.array([h_x, h_y, h_z])/mass
                I_bar = np.array([[I_xx, I_xy, I_xz],
                                  [I_xy, I_yy, I_yz],
                                  [I_xz, I_yz, I_zz]])
                
                # Update mass
                link.inertial.mass = mass

                # Update center of mass (com)
                link.inertial.origin.xyz = com.tolist()
                link.inertial.origin.rpy = [0.0, 0.0, 0.0]
                
                # Update inertia matrix
                I_c = I_bar - (mass * pin.skew(com) @ pin.skew(com).T)
                link.inertial.inertia.ixx = I_c[0, 0]
                link.inertial.inertia.ixy = I_c[0, 1]
                link.inertial.inertia.ixz = I_c[0, 2]
                link.inertial.inertia.iyy = I_c[1, 1]
                link.inertial.inertia.iyz = I_c[1, 2]
                link.inertial.inertia.izz = I_c[2, 2]
                
                # Update index
                idx += 10
        
        # Iterate over each joint and update its friction coeficients
        idx = 0
        for joint in robot.joints:
            if joint.type == "revolute":
                joint.dynamics.damping = b_v[idx]
                joint.dynamics.friction = b_c[idx]
                idx += 1
        
        # Save the updated URDF
        new_urdf_path = self.urdf_path.replace(".urdf", "_updated.urdf")
        with open(new_urdf_path, 'w') as f:
            f.write(robot.to_xml_string())
        print(f"Updated URDF saved at {new_urdf_path}")

    def compute_null_space_proj(self, cnt_schedule: np.ndarray) -> np.ndarray:
        # Returns null space projector, dim(nv, nv)
        J_c = self._compute_J_c(cnt_schedule)
        p = np.eye((self.nv)) - pinv(J_c) @ J_c
        return p

    def get_robot_mass(self):
        return self.robot_mass
    
    def get_num_links(self):
        return self.num_links
        
    def get_num_dof(self):
        return self.nv
    
    def get_bounding_ellipsoids(self):
        return self.bounding_ellipsoids
    
    def get_phi_nominal(self) -> np.ndarray:
        return self.phi_nom
    
    def get_physical_consistency(self, phi):
        # Returns the minimum eigenvalue of matrices in LMI constraints and trace(J@Q)
        # For phiysical consistency all values should be non-negative
        eigval_I_bar = np.zeros(self.num_links)
        eigval_I = np.zeros(self.num_links)      # Spatial body inertia
        eigval_J = np.zeros(self.num_links)      # Pseudo inertia
        trace_JQ = None
        
        for idx in range(self.num_links):
            j = idx * 10
            m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz = phi[j: j+10]
            h = np.array([h_x, h_y, h_z])

            # Inertia matrix (3x3)
            I_bar = np.array([
                [I_xx, I_xy, I_xz],
                [I_xy, I_yy, I_yz],
                [I_xz, I_yz, I_zz]
            ])

            # Spatial body inertia (6x6)
            I = np.zeros((6, 6), dtype=np.float32)
            I[:3, :3] = I_bar
            I[:3, 3:] = pin.skew(h)
            I[3:, :3] = pin.skew(h).T
            I[3:, 3:] = m * np.eye(3)

            # Pseudo inertia matrix (4x4)
            J = np.zeros((4, 4), dtype=np.float32)
            J[:3, :3] = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
            J[:3, 3] = h
            J[3, :3] = h
            J[3, 3] = m

            # Bounding ellipsoid constraint
            if self.bounding_ellipsoids is not None:
                trace_JQ = np.zeros(self.num_links)
                ellipsoid = self.bounding_ellipsoids[idx]
                semi_axes = ellipsoid["semi_axes"]
                center = ellipsoid["center"]

                Q = np.linalg.inv(np.diag(semi_axes) ** 2)

                Q_full = np.zeros((4, 4), dtype=np.float32)
                Q_full[:3, :3] = Q
                Q_full[:3, 3] = Q @ center
                Q_full[3, :3] = (Q @ center).T
                Q_full[3, 3] = 1.0 - center.T @ Q @ center

                trace_JQ[idx] = np.trace(J @ Q_full)

            # Eigenvalue checks
            eigval_I_bar[idx] = np.min(np.linalg.eigvalsh(I_bar))
            eigval_I[idx] = np.min(np.linalg.eigvalsh(I))
            eigval_J[idx] = np.min(np.linalg.eigvalsh(J))

        return eigval_I_bar, eigval_I, eigval_J, trace_JQ
         
    def get_regressor_matrix(self, q, dq, ddq):
        return pin.computeJointTorqueRegressor(self.rmodel, self.rdata, q, dq, ddq)
    
    def get_null_space_proj(self, cnt_schedule: np.ndarray) -> np.ndarray:
        # Returns null space projector, dim(nv, nv)
        J_c = self._compute_J_c(cnt_schedule)
        p = np.eye((self.nv)) - pinv(J_c) @ J_c
        return p

    def get_gen_force(self, tau, ee_force, cnt_schedule):
        J_c = self._compute_J_c(cnt_schedule)
        lamda = self._compute_lambda_c(ee_force, cnt_schedule)
        F = self.S.T @ tau + J_c.T @ lamda
        return F
    
    def get_cnt_force(self, force, cnt):
        J_c = self._compute_J_c(cnt)
        lamda = self._compute_lambda_c(force, cnt)
        F = J_c.T @ lamda
        return F
    
    def get_friction_regressors(self, dq):
        # Returns the viscous and Coulumb friction matrices
        B_v = self.S.T @ np.diag(dq[self.base_dof:])
        B_c = self.S.T @ np.diag(np.sign(dq[self.base_dof:]))
        return B_v, B_c
    
    # -------------------- Debugging and Visualization -------------------- #
    def print_tau_prediction_rmse(self, q, dq, ddq, torque, phi, param_name, cnt=None, b_v=None, b_c=None):
        # Shows RMSE of predicted torques based on phi parameters
        tau_pred = []
        tau_meas = []
        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        for i in range(q.shape[1]):
            self.update_fk(q[:, i], dq[:, i], ddq[:, i])
            Y = pin.computeJointTorqueRegressor(self.rmodel, self.rdata, q[:, i], dq[:, i], ddq[:, i])
            if cnt is not None:
                P = self.get_null_space_proj(cnt[:, i])
            else:
                P = np.eye(self.nv)
            if b_v is not None and b_c is not None:
                tau_prediction = P @ (Y @ phi 
                                - self.S.T @ (np.diag(b_v) @ dq[self.base_dof:, i] 
                                + np.diag(b_c) @ np.sign(dq[self.base_dof:, i])))
            else: 
                tau_prediction = P @ (Y @ phi)
            
            tau_pred.append(tau_prediction[self.base_dof:])
            tau_meas.append((P @ self.S.T @ torque[:, i])[self.base_dof:])
        tau_pred = np.vstack(tau_pred)
        tau_meas = np.vstack(tau_meas)
        
        error = tau_pred - tau_meas
        rmse_total = np.mean(np.square(np.linalg.norm(error, axis=1))) # overall RMSE
        joint_tau_rmse = np.sqrt(np.mean(np.square(error), axis=0)) # RMSE for each joint
        print("\n--------------------Torque Prediction Errors--------------------")
        print(f'RMSE for joint torques prediction using {param_name} parameters: total= {rmse_total}\nper_joints={joint_tau_rmse}')
    
    def print_inertial_params(self, prior, identified):
        total_m_prior = 0
        total_m_ident = 0
        self._cell_width = 13
        for i in range(self.num_links):
            expression = f'Inertial Parameters of "{self.link_names[i]}"'
            dash_length = (69 - len(expression)) // 2
            print(f'\n{"-"*dash_length} {expression} {"-"*(69-len(expression)-dash_length)}')
            print(
                f'|{"Parameter":<{self._cell_width}}|'
                f'{"A priori":<{self._cell_width}}|'
                f'{"Identified":<{self._cell_width}}|'
                f'{"Change":<{self._cell_width}}|'
                f'{"error %":<{self._cell_width}}|'
            )
            index = 10*i
            m_prior = prior[index]
            m_ident = identified[index]
            
            com_prior = prior[index+1: index+4]/m_prior
            com_ident = identified[index+1: index+4]/m_ident
            
            inertia_prior = prior[index+4:index+10]
            inertia_ident = identified[index+4:index+10]
            
            self._print_table("mass (kg)", m_prior, m_ident)
            self._print_table("c_x (m)", com_prior[0], com_ident[0])
            self._print_table("c_y (m)", com_prior[1], com_ident[1])
            self._print_table("c_z (m)", com_prior[2], com_ident[2])
            self._print_table("I_xx (kg.m^2)", inertia_prior[0], inertia_ident[0])
            self._print_table("I_xy (kg.m^2)", inertia_prior[1], inertia_ident[1])
            self._print_table("I_xz (kg.m^2)", inertia_prior[2], inertia_ident[2])
            self._print_table("I_yy (kg.m^2)", inertia_prior[3], inertia_ident[3])
            self._print_table("I_yz (kg.m^2)", inertia_prior[4], inertia_ident[4])
            self._print_table("I_zz (kg.m^2)", inertia_prior[5], inertia_ident[5])
            
            total_m_prior += m_prior
            total_m_ident += m_ident
        print(f'\nRobot total mass: {total_m_prior} ---- Identified total mass: {total_m_ident}')
    
    def _print_table(self, description, prior, ident):
        precision = 6
        change = ident - prior
        error = np.divide(change, np.abs(prior), where=prior!=0) * 100
        error = np.where(np.abs(prior) <= 1e-8, np.nan, error)
        print(
        f'|{description:<{self._cell_width}}|'
        f'{prior:>{self._cell_width}.{precision}f}|'
        f'{ident:>{self._cell_width}.{precision}f}|'
        f'{change:>{self._cell_width}.{precision}f}|'
        f'{error:>{self._cell_width}.{1}f}|'
        )