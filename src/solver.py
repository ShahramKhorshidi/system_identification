import numpy as np
import cvxpy as cp


class Solver():
    def __init__(self, regressor, tau_vec, num_links, phi_prior, total_mass, bounding_ellipsoids, B_v=None, B_c=None):
        self._Y = regressor
        self._tau = tau_vec
        self._nx = self._Y.shape[1]
        self._num_links = num_links
        self._num_inertial_params = self._Y.shape[1] // self._num_links
        
        self._phi_prior = phi_prior  # Prior inertial parameters
        self.total_mass = total_mass
        self._bounding_ellipsoids = bounding_ellipsoids
        
        # Initialize optimization variables and problem to use solvers from cp
        self._x = cp.Variable(self._nx, value=phi_prior)
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
    
    ## --------- Unconstrained Solver --------- ##
    def solve_llsq_svd(self):        
        """
        Solve llsq using Singular Value Decomposition (SVD).
        """
        U, Sigma, VT = np.linalg.svd(self._Y, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self._tau

    ## --------- Constrained Solver (LMI) --------- ##
    def _construct_pseudo_inertia_matrix(self, phi):
        # Retunrs the pseudo inertia matrix (J: 4x4)
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
        Q = np.linalg.inv(np.diag(semi_axes)**2)
        Qc = Q @ center
        Q_full = np.vstack([np.hstack([-Q, Qc[:, np.newaxis]]), np.append(Qc, 1 - center.T @ Qc)])
        return Q_full
    
    def _construct_com_constraint_matrix(self, phi, semi_axes, center):
        com_constraint = np.zeros((4,4), dtype=np.float32)
        mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi.value
        h = np.array([h_x, h_y, h_z])
        Qs = np.diag(semi_axes)**2
        com_constraint[0, 0] = mass
        com_constraint[0, 1:] = h.T - mass * center.T
        com_constraint[1:, 0] = h - mass * center
        com_constraint[1:, 1:] = mass * Qs
        com_constraint_param = cp.Parameter(com_constraint.shape, value=com_constraint, symmetric=True)
        return com_constraint_param
    
    def _pullback_metric(self, phi):
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
    
    def solve_fully_consistent(self, lambda_reg=1e-1, epsillon=1e-3, max_iter=20000, reg_type="entropic"):
        """
        Solve constrained least squares problem as LMI. Ensuring physical fully-consistency.
        """
        mass_sum = 0  # To accumulate the total mass
        regularization_term = 0
        self._constraints = []
        
        # Iterating over the robot links
        for idx in range(0, self._num_links):
            # Extracting the inertial parameters of the link idx (phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz])
            j = idx * self._num_inertial_params
            phi_idx = self._x[j: j+self._num_inertial_params]
            phi_prior_idx = self._phi_prior[j: j+self._num_inertial_params]
            ellipsoid_params_idx = self._bounding_ellipsoids[idx]
            
            # Mass
            mass_idx = phi_idx[0]
            mass_sum += mass_idx
            
            # Add pseudo inertia matrix (J:4x4) constraint
            J = self._construct_pseudo_inertia_matrix(phi_idx)
            self._constraints.append(J >> cp.Constant(0)) # Positive definite constraint
            
            # Add the CoM constraint
            com_constraint = self._construct_com_constraint_matrix(phi_idx, ellipsoid_params_idx['semi_axes'], ellipsoid_params_idx['center'])
            self._constraints.append(com_constraint >= cp.Constant(0))
            
            # Add the bounding ellipsoid constraint
            Q_ellipsoid = self._construct_ellipsoid_matrix(ellipsoid_params_idx['semi_axes'], ellipsoid_params_idx['center'])
            self._constraints.append(cp.trace(J @ Q_ellipsoid) >= cp.Constant(0))
            
            # Regularization terms
            if reg_type=="constant_pullback":
                # Constant pullback approximation
                M = self._pullback_metric(phi_prior_idx)
                phi_diff_idx = phi_idx - phi_prior_idx
                regularization_term += (1/2) * cp.quad_form(phi_diff_idx, M)
            elif reg_type=="entropic":
                # Entropic (Bregman) divergence
                J_prior = self._construct_pseudo_inertia_matrix(phi_prior_idx)
                U, Sigma, VT = np.linalg.svd(J_prior.value, full_matrices=True)
                Sigma_inv = np.linalg.pinv(np.diag(Sigma))
                # Solve for : J_prior @ X = J
                X = VT.T @ Sigma_inv @ U.T @ J.value
                regularization_term += -cp.log_det(J) + cp.log(np.linalg.det(J_prior.value)) + cp.trace(X) - 4
        # Regularization based on Euclidean distance from phi_prior
        if reg_type=="euclidean":
            phi_diff_all = self._x - self._phi_prior
            regularization_term = cp.quad_form(phi_diff_all, np.eye(self._x.shape[0]))
        
        # Add the total mass constraint
        self._constraints.append(mass_sum == self.total_mass)
        
        # Add objective function and instantiate problem
        if self._identify_fric:
            self._constraints.append(self._b_v >= cp.Constant(0))
            self._constraints.append(self._b_c >= cp.Constant(0))
            error = cp.sum_squares(self._Y @ self._x + self._B_v @ self._b_v + self._B_c @ self._b_c - self._tau) / self._Y.shape[0]
        else:
            error = cp.sum_squares(self._Y @ self._x - self._tau) / self._Y.shape[0]
        
        self._objective = cp.Minimize(error + lambda_reg * regularization_term)
        self._problem = cp.Problem(self._objective, self._constraints)
        
        # Check if the problem is DPP compliant
        if self._problem.is_dcp(dpp=True):
            self._problem.solve(solver=cp.SCS, eps=epsillon, max_iters=max_iter, warm_start=True, verbose=True)
        else:
            raise ValueError("The problem is not DPP compliant.")

        if self._problem.status == cp.OPTIMAL or self._problem.status == cp.OPTIMAL_INACCURATE:
            print("########################################")
            # Optimal value of the objective function
            print("Optimal value:", self._problem.value)
            # Solver-specific information
            solver_info = self._problem.solver_stats
            print("Solver time (seconds):", solver_info.solve_time)
            print("Setup time (seconds):", solver_info.setup_time)
            print("Number of iterations:", solver_info.num_iters)
            print("########################################")
            
            if self._identify_fric:
                # Return the value of the decision variables
                return self._x.value, self._b_v.value, self._b_c.value
            else:
                return self._x.value
        else:
            print("The problem did not solve to optimality. Status:", self._problem.status)
            raise ValueError("The problem did not solve to optimality.")