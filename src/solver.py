import numpy as np
import cvxpy as cp


class Solver():
    def __init__(self, regressor, tau_vec, num_links, phi_prior, total_mass, bounding_ellipsoids, B_v=None, B_c=None):
        self._Y = regressor
        self._tau = tau_vec
        self._nx = self._Y.shape[1]
        self._num_samples = self._Y.shape[0]
        self._num_links = num_links
        self._num_inertial_params = self._Y.shape[1] // self._num_links
        
        self._phi_prior = phi_prior  # Prior inertial parameters
        self.total_mass = total_mass
        self._bounding_ellipsoids = bounding_ellipsoids
        
        # Initialize optimization variables and problem to use solvers from cp
        self._phi = cp.Variable(self._nx, value=phi_prior)
        self._identify_fric = (B_v is not None) and (B_c is not None)
        if self._identify_fric:
            self._B_v = B_v
            self._B_c = B_c
            self.ndof = B_v.shape[1]
            self._b_v = cp.Variable(self.ndof) # Viscous friction coefficient (Nm / (rad/s))
            self._b_c = cp.Variable(self.ndof) # Coulomb friction coefficient (Nm)
        self._objective = None
        self._constraints = []
        self._problem = None
    
    # -------------- Unconstrained Solver -------------- #
    def solve_llsq_svd(self):        
        """
        Solve llsq using Singular Value Decomposition (SVD).
        """
        U, Sigma, VT = np.linalg.svd(self._Y, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self._tau

    # ------------ Constrained Solver (LMI) ------------ #
    def _construct_spatial_body_inertia_matrix(self, phi):
        # Retunrs the spatial body inertia matrix (S:6x6) as a cvxpy expression
        m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi
        S = cp.vstack([
            cp.hstack([I_xx, I_xy, I_xz, 0   , -h_z, h_y ]),
            cp.hstack([I_xy, I_yy, I_yz, h_z , 0   , -h_x]),
            cp.hstack([I_xz, I_yz, I_zz, -h_y, h_x , 0   ]),
            cp.hstack([0   , h_z , -h_y, m   , 0   , 0   ]),
            cp.hstack([-h_z, 0   , h_x ,0    , m   , 0   ]),
            cp.hstack([h_y , -h_x, 0   ,0    , 0   , m   ])
        ])
        return S
        
    def _construct_pseudo_inertia_matrix(self, phi):
        # Retunrs the pseudo inertia matrix (J:4x4) as a cvxpy expression
        mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi
        trace = (1/2) * (I_xx + I_yy + I_zz)
        pseudo_inertia_matrix = cp.vstack([
            cp.hstack([trace-I_xx, -I_xy     , -I_xz     , h_x ]),
            cp.hstack([-I_xy     , trace-I_yy, -I_yz     , h_y ]),
            cp.hstack([-I_xz     , -I_yz     , trace-I_zz, h_z ]),
            cp.hstack([h_x       , h_y       , h_z       , mass])
        ])
        return pseudo_inertia_matrix

    def _construct_ellipsoid_matrix(self, semi_axes, center):
        # Returns the bounding ellipsoid matrix (Q:4x4) as a numpy array
        Q_full = np.zeros((4,4), dtype=np.float32)
        Q = np.linalg.inv(np.diag(semi_axes)**2)
        Q_full[:3, :3] = Q
        Q_full[:3, 3] = Q @ center
        Q_full[3, :3] = (Q @ center).T
        Q_full[3, 3] = 1 - (center.T @ Q @ center)
        return Q_full
    
    def _construct_com_constraint_matrix(self, phi, semi_axes, center):
        # Retunrs the CoM constraint matrix (com:4x4) as a cvxpy expression
        mass = phi[0]
        h = phi[1:4]
        Qs = np.diag(semi_axes) ** 2
        
        # Construct the individual blocks of the constraint matrix
        top_left = cp.reshape(mass, (1, 1))  # Scalar reshaped to (1,1)
        top_right = cp.reshape(h - mass * center, (1, 3))  # (3,) array reshaped to (1,3)
        bottom_left = top_right.T  # Transpose to get (3,1)
        bottom_right = mass * Qs  # Already (3,3)
        
        com_constraint = cp.bmat([
            [top_left   , top_right   ],
            [bottom_left, bottom_right]
        ])
        return com_constraint
    
    def _pullback_metric(self, phi):
        # Returns the approximation of Riemannian distance metric, (M:10x10) as a numpy array
        M = np.zeros((self._num_inertial_params, self._num_inertial_params))
        P = self._construct_pseudo_inertia_matrix(phi).value
        P_inv = np.linalg.inv(P)
        
        for i in range(10):
            for j in range(10):
                v_i = np.zeros(10)
                v_j = np.zeros(10)
                v_i[i] = 1
                v_j[j] = 1
                V_i = self._construct_pseudo_inertia_matrix(v_i).value
                V_j = self._construct_pseudo_inertia_matrix(v_j).value
                M[i, j] = np.trace(P_inv @ V_i @ P_inv @ V_j)
        
        # Ensure M is symmetric
        M = (M + M.T) / 2
        
        # Ensure M is positive semi-definite
        eigenvalues = np.linalg.eigvals(M)
        if np.any(eigenvalues < 0):
            shift = - np.min(eigenvalues) + 1e-5
            M = M + shift * np.eye(M.shape[0])
        min_eigenvalue = np.min(np.linalg.eigvals(M))
        assert min_eigenvalue > 0, f"Matrix is not positive definite. Minimum eigenvalue: {min_eigenvalue}"
        return M
    
    def solve_fully_consistent(self, lambda_reg=1e-1, tol=1e-10, max_iters=1000, reg_type="constant_pullback"):
        """
        Solve constrained least squares problem as LMI. Ensuring full physical consistency.
        """
        mass_sum = 0  # To accumulate the total mass
        regularization_term = 0
        self._constraints = []
        
        # Iterating over the robot links
        for idx in range(0, self._num_links):
            # Extracting the inertial parameters of the link idx (phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz])
            j = idx * self._num_inertial_params
            phi_idx = self._phi[j: j+self._num_inertial_params]
            phi_prior_idx = self._phi_prior[j: j+self._num_inertial_params]
            ellipsoid_params_idx = self._bounding_ellipsoids[idx]
            
            # Mass constraint
            self._constraints.append(phi_idx[0] >= 0)
            mass_sum += phi_idx[0]
            
            # Compute pseudo inertia matrix (J:4x4) and add the constraint
            J = self._construct_pseudo_inertia_matrix(phi_idx)
            epsilon = 1e-6
            J_reg = J + epsilon * cp.Constant(np.eye(J.shape[0])) # Regularize to ensure J is strictly positive definite
            self._constraints.append(J_reg >> 0)
            
            # Compute CoM matrix (com:4x4) and add the constraint
            com = self._construct_com_constraint_matrix(phi_idx, ellipsoid_params_idx['semi_axes'], ellipsoid_params_idx['center'])
            com_reg = com + epsilon * cp.Constant(np.eye(com.shape[0]))
            self._constraints.append(com_reg >> 0)
            
            # Compute ellipsoid matrix (Q:4x4) and add the density realizability constraint
            Q_ellipsoid = self._construct_ellipsoid_matrix(ellipsoid_params_idx['semi_axes'], ellipsoid_params_idx['center'])
            self._constraints.append(cp.trace(J @ Q_ellipsoid) >= 0)
            
            # Regularization terms based on coordinate-invariant Riemannian geometric
            if reg_type=="constant_pullback":
                # Constant pullback approximation
                M = self._pullback_metric(phi_prior_idx)
                phi_diff_idx = phi_idx - phi_prior_idx
                regularization_term += (1/2) * cp.quad_form(phi_diff_idx, M)
            elif reg_type=="entropic":
                # TODO: with this regularization the problem doesn't converge!
                # Entropic (Bregman) divergence
                J_prior = self._construct_pseudo_inertia_matrix(phi_prior_idx)
                U, Sigma, VT = np.linalg.svd(J_prior.value, full_matrices=True)
                Sigma_inv = np.linalg.pinv(np.diag(Sigma))
                # Solve for : J_prior @ X = J
                X = VT.T @ Sigma_inv @ U.T @ J
                regularization_term += -cp.log_det(J) + cp.log(np.linalg.det(J_prior.value)+1e-5) + cp.trace(X) - 4
        
        # Regularization based on Euclidean distance from phi_prior
        if reg_type=="euclidean":
            phi_diff_all = self._phi - self._phi_prior
            regularization_term = cp.quad_form(phi_diff_all, np.eye(self._nx))
        
        # Add the total mass constraint
        self._constraints.append(mass_sum == self.total_mass)
        
        # Add objective function and instantiate problem
        if self._identify_fric:
            self._constraints.append(self._b_v >= 0)
            self._constraints.append(self._b_c >= 0)
            error = self._Y @ self._phi + self._B_v @ self._b_v + self._B_c @ self._b_c - self._tau
        else:
            error = self._Y @ self._phi - self._tau
        
        self._objective = cp.Minimize( (1/2) * cp.sum_squares(error) / self._num_samples + lambda_reg * regularization_term)
        self._problem = cp.Problem(self._objective, self._constraints)
        
        # Ensure the problem is DCP compliant
        try:
            self._problem.solve(solver=cp.MOSEK,
                                mosek_params = {
                                    'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_PRIMAL',
                                    'MSK_DPAR_INTPNT_TOL_REL_GAP': tol,
                                    'MSK_DPAR_OPTIMIZER_MAX_TIME':  100.0,
                                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                    },
                                verbose=True)
        except Exception as e:
            print(e)
        
        if self._problem.status == cp.OPTIMAL or self._problem.status == cp.OPTIMAL_INACCURATE:
            return self._phi.value
        else:
            print("The problem did not solve to optimality. Status:", self._problem.status)
            raise ValueError("The problem did not solve to optimality.")