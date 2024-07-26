import numpy as np
import cvxpy as cp


class Solver():
    def __init__(self, A_matrix, b_vec, num_links, phi_prior, total_mass, bounding_ellipsoids):
        self._A = A_matrix
        self._b = b_vec
        self._nx = self._A.shape[1]
        self._num_links = num_links
        self._num_inertial_params = self._A.shape[1] // self._num_links
        
        self._bounding_ellipsoids = bounding_ellipsoids
        self._phi_prior = phi_prior  # Prior inertial parameters
        self.total_mass = total_mass
        
        # Initialize optimization variables and problem to use solvers from cp
        self._x = cp.Variable(self._nx, value=phi_prior)
        self._objective = None
        self._constraints = []
        self._problem = None
    
    ## --------- Unconstrained Solver --------- ##
    def solve_llsq_svd(self):        
        """
        Solve llsq using Singular Value Decomposition (SVD).
        """
        U, Sigma, VT = np.linalg.svd(self._A, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self._b

    ## --------- Constrained Solver (LMI) --------- ##
    def _construct_pseudo_inertia_matrix(self, phi):
        # Retunrs the pseudo inertia matrix (J: 4x4)
        mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi
        trace = 0.5 * (I_xx + I_yy + I_zz)
        pseudo_inertia_matrix = cp.vstack([
            cp.hstack([trace-I_xx, -I_xy     , -I_xz     ,  h_x]),
            cp.hstack([-I_xy     , trace-I_yy, -I_yz     ,  h_y]),
            cp.hstack([-I_xz     , -I_yz     , trace-I_zz,  h_z]),
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
    
    def solve_fully_consistent(self, lambda_reg=1e-1, epsillon=1e-3, max_iter=20000, reg_type="euclidean"):
        """
        Solve constrained least squares problem as LMI. Ensuring physical Fully-consistency.
        """
        mass_sum = 0  # To accumulate the total mass
        regularization_term = 0
        self._constraints = []
        
        for idx in range(0, self._nx, self._num_inertial_params):
            # Extracting the inertial parameters (phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz])
            phi_idx = self._x[idx: idx+self._num_inertial_params]
            phi_prior_idx = self._phi_prior[idx: idx+self._num_inertial_params]
            ellipsoid_params = self._bounding_ellipsoids[idx // self._num_inertial_params]
            
            # Mass
            mass = phi_idx[0]
            mass_sum += mass
            
            # Add pseudo inertia matrix (J:4x4) constraint
            J = self._construct_pseudo_inertia_matrix(phi_idx)
            self._constraints.append(J >> cp.Constant(0)) # Positive definite constraint
            
            # Add the CoM constraint
            com_constraint = self._construct_com_constraint_matrix(phi_idx, ellipsoid_params['semi_axes'], ellipsoid_params['center'])
            self._constraints.append(com_constraint >= cp.Constant(0))
            
            # Add the bounding ellipsoid constraint
            Q_ellipsoid = self._construct_ellipsoid_matrix(ellipsoid_params['semi_axes'], ellipsoid_params['center'])
            self._constraints.append(cp.trace(J @ Q_ellipsoid) >= cp.Constant(0))
            
            # Regularization terms
            if reg_type=="constant_pullback":
                # Constant pullback approximation
                M = self._pullback_metric(phi_prior_idx)
                phi_diff_idx = phi_idx - phi_prior_idx
                regularization_term += 0.5 * cp.quad_form(phi_diff_idx, M)
            elif reg_type=="entropic":
                # Entropic (Bregman) divergence
                trace_prior = 0.5 * (phi_prior_idx[4] + phi_prior_idx[7] + phi_prior_idx[9])
                J_prior = self._construct_pseudo_inertia_matrix(phi_prior_idx)
                regularization_term += (-cp.log_det(J) + cp.log_det(J_prior) 
                                       + cp.trace(cp.inv_pos(J_prior) @ J) - 4)
        # Regularization based on Euclidean distance from phi_prior
        if reg_type=="euclidean":
            phi_diff_all = self._x - self._phi_prior
            regularization_term = cp.quad_form(phi_diff_all, np.eye(self._x.shape[0]))
        
        # Add the total mass constraint
        self._constraints.append(mass_sum == self.total_mass)
        
        # Add objective function and instantiate problem
        self._objective = cp.Minimize(
            cp.sum_squares(self._A @ self._x - self._b)/ self._A.shape[0] + lambda_reg * regularization_term
        )
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
            
            # Return the value of the decision variable
            return self._x.value
        else:
            print("The problem did not solve to optimality. Status:", self._problem.status)
            raise ValueError("The problem did not solve to optimality.")