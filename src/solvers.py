import numpy as np
import cvxpy as cp


class Solvers():
    def __init__(self, A_matrix, b_vec, num_links, phi_prior, bounding_ellipsoids):
        self._A = A_matrix
        self._b = b_vec
        self._nx = self._A.shape[1]
        self._num_links = num_links
        self._num_inertial_param = int(self._A.shape[1] / self._num_links)
        
        self._bounding_ellipsoids = bounding_ellipsoids
        self._phi_prior = phi_prior  # Prior inertial parameters
        
        # Initialize optimization variables and problem to use solvers from cp
        self._x = cp.Variable(self._nx, value=phi_prior)
        self._objective = None
        self._constraints = []
        self._problem = None
    
    ## --------- Unconstrained Solvers --------- ##
    def normal_equation(self, lambda_reg=0.1):
        """
        Solve llsq using the normal equations with regularization.
        """
        # lambda_reg is regularization parameter
        I = np.eye(self._nx)
        return np.linalg.inv(self._A.T @ self._A + lambda_reg * I) @ self._A.T @ self._b
    
    def conjugate_gradient(self, tol=1e-6, max_iter=1000):
        """
        Solve the normal equations using the Conjugate Gradient method.
        When A is full column rank.
        """
        # Compute A.T@A and A.T@b
        AtA = self._A.T @ self._A
        Atb = self._A.T @ self._b
        
        # Initialize x, r, p
        x = np.zeros(self._nx)
        r = Atb - AtA @ x
        p = r.copy()
        rs_old = np.dot(r.T, r)
        
        for i in range(max_iter):
            Ap = AtA @ p
            alpha = rs_old / np.dot(p.T, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r.T, r)
            
            if np.sqrt(rs_new) < tol:
                print(f"Converged in {i+1} iterations")
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            
        return x
    
    def solve_llsq_svd(self):        
        """
        Solve llsq using Singular Value Decomposition (SVD).
        """
        U, Sigma, VT = np.linalg.svd(self._A, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self._b
    
    def wighted_llsq(self, weights=None):
        """
        Solve weighted least squares using cvxpy.
        """
        if weights is None:
            weights = np.ones(self._A.shape[0])
        
        # Define the weighted least squares objective function
        residuals = self._A @ self._x - self._b
        self._objective = cp.Minimize(cp.sum_squares(cp.multiply(weights, residuals)))
        self._problem = cp.Problem(self._objective)
        self._problem.solve()
        return self._x.value
    
    def ridge_regression(self, lambda_reg=0.1, x_init=None, warm_start=False):
        """
        Solve ridge regression using cvxpy with optional warm start.
        """
        # Set the initial value of the decision variable
        if x_init is None:
            x_init = np.zeros(self._nx)
        self._x.value = x_init
        
        self._objective = cp.Minimize(cp.sum_squares(self._A @ self._x - self._b) + lambda_reg * cp.norm(self._x, 2))
        self._problem = cp.Problem(self._objective)
        self._problem.solve(solver=cp.SCS, warm_start=warm_start)
        return self._x.value

    ## --------- Constrained Solvers (LMI) --------- ##
    def _construct_spatial_inertia_matrix(self, phi):
        # Returns the spatila body inertia matrix (6x6)
        mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi
        spatial_inertia_matrix = cp.vstack([
            cp.hstack([I_xx, I_xy, I_xz, 0   , -h_z , h_y ]),
            cp.hstack([I_xy, I_yy, I_yz, h_z ,  0   , -h_x]),
            cp.hstack([I_xz, I_yz, I_zz, -h_y,  h_x ,  0  ]),
            cp.hstack([0   , h_z , -h_y, mass,  0   ,  0  ]),
            cp.hstack([-h_z, 0   , h_x , 0   ,  mass,  0  ]),
            cp.hstack([h_y , -h_x, 0   , 0   ,  0   , mass])
        ])
        return spatial_inertia_matrix
    
    def solve_semi_consistent(self, total_mass, lambda_reg=1e-1):
        """
        Solve constrained least squares problem as LMI. Ensuring physical Semi-consistency.
        """
        mass_sum = 0  # To accumulate the total mass
        self._constraints = []  # Ensure constraints list is fresh
        
        for j in range(0, self._nx, self._num_inertial_param):
            # Extracting the Inertial parameters (phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz])
            phi_j = self._x[j:j+self._num_inertial_param]

            # Mass
            mass = phi_j[0]
            mass_sum += mass
            
            # Spatial inertia matrix (I: 6x6)
            spatial_inertia_matrix = self._construct_spatial_inertia_matrix(phi_j)
            self._constraints.append(spatial_inertia_matrix >> 0)  # Positive definite constraint
        
        # Add the total mass constraint
        self._constraints.append(mass_sum == total_mass)
        
        self._objective = cp.Minimize(
            cp.sum_squares(self._A @ self._x - self._b) / self._A.shape[0]  + lambda_reg * cp.norm(self._x, 2)
        )
        
        self._problem = cp.Problem(self._objective, self._constraints)
        self._problem.solve(solver=cp.SCS, verbose=True, eps=1e-3, max_iters=20000)
        return self._x.value
    
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
        mass, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi
        h = cp.vstack([h_x, h_y, h_z])  # Ensure h is a CVXPY variable
        Qs = np.diag(semi_axes)**2
        Q_cp = cp.Parameter((3, 3), value=Qs)
        center_cp = cp.Parameter((3, 1), value=center.reshape(-1, 1))  # Center as a column vector
        
        mass_cp = cp.reshape(mass, (1, 1))
        top_left = mass_cp
        top_right = h.T - mass_cp * center_cp.T # This should be a row vector
        bottom_left = h - mass_cp * center_cp # This should be a column vector
        bottom_right = mass_cp * Q_cp # This should be a 3x3 matrix
        
        top_row = cp.hstack([top_left, top_right])
        bottom_row = cp.hstack([bottom_left, bottom_right])
        com_constraint = cp.vstack([top_row, bottom_row])

        return com_constraint
    
    def _pullback_metric(self, phi):
        M = np.zeros((10, 10))
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
            M = M - np.min(eigenvalues) * np.eye(M.shape[0])
            
        return M
    
    def solve_fully_consistent(self, total_mass, lambda_reg=1e-1, use_const_pullback_approx=True):
        """
        Solve constrained least squares problem as LMI. Ensuring physical Fully-consistency.
        """
        mass_sum = 0  # To accumulate the total mass
        bregman_divergence = 0  # Initialize Bregman divergence
        self._constraints = []  # Ensure constraints list is fresh
        
        for j in range(0, self._nx, self._num_inertial_param):
            # Extracting the inertial parameters (phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz])
            phi_j = self._x[j: j+self._num_inertial_param]
            phi_prior_j = self._phi_prior[j: j+self._num_inertial_param]
            ellipsoid_params = self._bounding_ellipsoids[j // self._num_inertial_param]
            
            # Mass
            mass = phi_j[0]
            mass_sum += mass
            
            # Add pseudo inertia matrix (J:4x4) constraint
            pseudo_inertia_matrix = self._construct_pseudo_inertia_matrix(phi_j)
            self._constraints.append(pseudo_inertia_matrix >> 0) # Positive definite constraint
            
            # Add the CoM constraint
            com_constraint = self._construct_com_constraint_matrix(phi_j, ellipsoid_params['semi_axes'], ellipsoid_params['center'])
            self._constraints.append(com_constraint >> 0)
            
            # Add the bounding ellipsoid constraint
            Q_ellipsoid = self._construct_ellipsoid_matrix(ellipsoid_params['semi_axes'], ellipsoid_params['center'])
            self._constraints.append(cp.trace(pseudo_inertia_matrix @ Q_ellipsoid) >= 0)
            
            # # Bregman divergence regularization term
            # if use_const_pullback_approx:
            #     M = self._pullback_metric(phi_prior_j)
            #     phi_diff = phi_j - phi_prior_j
            #     bregman_divergence += 0.5 * cp.quad_form(phi_diff, M)
            # else:
            #     trace_prior = 0.5 * (phi_prior_j[4] + phi_prior_j[7] + phi_prior_j[9])
            #     J_prior = self._construct_pseudo_inertia_matrix(phi_prior_j)
            #     bregman_divergence += (-cp.log_det(pseudo_inertia_matrix) + cp.log_det(J_prior) 
            #                            + cp.trace(cp.inv_pos(J_prior) @ pseudo_inertia_matrix) - 4)
        
        # Add the total mass constraint
        self._constraints.append(mass_sum == total_mass)
        
        # Regularization term based on Euclidean distance from phi_prior
        phi_diff_all = self._x - self._phi_prior
        euclidean_reg = cp.quad_form(phi_diff_all, np.eye(len(self._phi_prior)))
        
        self._objective = cp.Minimize(
            cp.sum_squares(self._A @ self._x - self._b)/ self._A.shape[0] + lambda_reg * euclidean_reg
        )
        
        self._problem = cp.Problem(self._objective, self._constraints)
        print(f"################ prob4 is DCP: {self._problem.is_dcp(dpp=True)}")
        self._problem.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=30000)
        print("status:", self._problem.status)
        return self._x.value