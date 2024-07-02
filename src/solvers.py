import numpy as np
import cvxpy as cp


class Solvers():
    def __init__(self, A_matrix, b_vec, phi_prior):
        self.A = A_matrix
        self.b = b_vec
        self.nx = self.A.shape[1]
        self._num_inertial_param = 10
        self.num_links = self.A.shape[1] // self._num_inertial_param
        
        self.phi_prior = phi_prior  # Prior inertial parameters
        self.use_const_pullback_approx = 1
        
        # Initialize optimization variables and problem to use solvers from cp
        self.x = cp.Variable(self.nx)
        self.objective = None
        self.constraints = []
        self.problem = None
    
    def normal_equation(self, lambda_reg=0.1):
        """
        Solve llsq using the normal equations with regularization.
        """
        # lambda_reg is regularization parameter
        I = np.eye(self.nx)
        return np.linalg.inv(self.A.T @ self.A + lambda_reg * I) @ self.A.T @ self.b
    
    def conjugate_gradient(self, tol=1e-6, max_iter=1000):
        """
        Solve the normal equations using the Conjugate Gradient method.
        When A is full column rank.
        """
        # Compute A.T@A and A.T@b
        AtA = self.A.T @ self.A
        Atb = self.A.T @ self.b
        
        # Initialize x, r, p
        x = np.zeros(self.nx)
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
        U, Sigma, VT = np.linalg.svd(self.A, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self.b
    
    def wighted_llsq(self, weights=None):
        """
        Solve weighted least squares using cvxpy.
        """
        if weights is None:
            weights = np.ones(self.A.shape[0])
        
        # Define the weighted least squares objective function
        residuals = self.A @ self.x - self.b
        self.objective = cp.Minimize(cp.sum_squares(cp.multiply(weights, residuals)))
        self.problem = cp.Problem(self.objective)
        self.problem.solve()
        return self.x.value
    
    def ridge_regression(self, lambda_reg=0.1, x_init=None, warm_start=False):
        """
        Solve ridge regression using cvxpy with optional warm start.
        """
        # Set the initial value of the decision variable
        if x_init is None:
            x_init = np.zeros(self.nx)
        self.x.value = x_init
        
        self.objective = cp.Minimize(cp.sum_squares(self.A @ self.x - self.b) + lambda_reg * cp.norm(self.x, 2))
        self.problem = cp.Problem(self.objective)
        self.problem.solve(solver=cp.SCS, warm_start=warm_start)
        return self.x.value

    def _construct_spatial_inertia_matrix(self, phi):
        # Returns the spatila body inertia (6x6)
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
        Solve constrained least squares problem. Ensuring physical Semi-consistency.
        """
        self.objective = cp.Minimize(cp.sum_squares(self.A @ self.x - self.b) + lambda_reg * cp.norm(self.x, 2))
        
        # phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz] (for each link)
        mass_sum = 0  # To accumulate the total mass
        for j in range(0, self.nx, self._num_inertial_param):
            # Extracting the Inertial parameters
            phi_j = self.x[j:j+self._num_inertial_param]
            
            # Mass
            mass = phi_j[0]
            mass_sum += mass
            
            # Spatial inertia matrix (I: 6x6)
            spatial_inertia_matrix = self._construct_spatial_inertia_matrix(phi_j)
            self.constraints.append(spatial_inertia_matrix >> 0)  # Positive definite constraint
        
        # Add the total mass constraint
        self.constraints.append(mass_sum == total_mass)

        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem.solve(solver=cp.SCS, verbose=True, eps=1e-3, max_iters=20000)
        return self.x.value
    
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
    
    def _pullback_metric(self, params):
        M = np.zeros((10, 10))
        P = self._construct_pseudo_inertia_matrix(params).value
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
        
        return M
    
    def solve_fully_consistent(self, total_mass, lambda_reg=1e-1):
        """
        Solve constrained least squares problem. Ensuring physical Fully-consistency.
        """
        # For each link: phi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]
        mass_sum = 0  # To accumulate the total mass
        
        for j in range(0, self.nx, self._num_inertial_param):
            # Extracting the inertial parameters
            phi_j = self.x[j: j+self._num_inertial_param]
            phi_prior_j = self.phi_prior[j: j+self._num_inertial_param]
            
            # Mass
            mass = phi_j[0]
            mass_sum += mass
            
            # Pseudo inertia matrix (J: 4x4)
            pseudo_inertia_matrix = self._construct_pseudo_inertia_matrix(phi_j)
            self.constraints.append(pseudo_inertia_matrix >> 0) # Positive definite constraint
            
            # Bregman Divergence Term
            if self.use_const_pullback_approx:
                trace_prior = 0.5 * (phi_prior_j[4] + phi_prior_j[7] + phi_prior_j[9])
                J_prior = self._construct_pseudo_inertia_matrix(phi_prior_j)
                bregman_divergence += -cp.log_det(pseudo_inertia_matrix) + cp.log_det(J_prior) + cp.trace(cp.inv_pos(J_prior) @ pseudo_inertia_matrix) - 4
            else:
                M = self._pullback_metric(phi_prior_j)
                bregman_divergence += 0.5 * cp.quad_form(phi_j - phi_prior_j, M)
        
        # Add the total mass constraint
        self.constraints.append(mass_sum == total_mass)
        
        self.objective = cp.Minimize(
            cp.sum_squares(self.A @ self.x - self.b)/ self.A.shape[0] + lambda_reg * bregman_divergence
        )
        
        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=30000)
        return self.x.value