import numpy as np
import cvxpy as cp


class Solvers():
    def __init__(self, A_matrix, b_vec):
        self.A = A_matrix
        self.b = b_vec
        self.nx = self.A.shape[1]
        
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