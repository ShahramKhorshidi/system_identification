import numpy as np
import casadi as ca


class Solver:
    def __init__(self, regressor, tau_vec):
        self._Y = regressor
        self._tau = tau_vec
        self._nx = self._Y.shape[1]
        self._num_samples = self._Y.shape[0]

        # Initialize optimization variables and problem to use solvers from CasADi
        self._x = ca.MX.sym('x', self._nx)
        
        self._objective = None
        self._solver = None
        self._constraints = []
        self._parameters = []
    
    ## --------- Unconstrained Solver --------- ##
    def solve_llsq_svd(self):        
        """
        Solve llsq using Singular Value Decomposition (SVD).
        """
        U, Sigma, VT = np.linalg.svd(self._Y, full_matrices=False)
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        A_psudo = VT.T @ Sigma_inv @ U.T
        return A_psudo@self._tau    
    
    def solve_fully_consistent(self, lambda_reg=1e-1, epsillon=1e-3, max_iter=20000, reg_type="entropic"):
        error = ca.sumsqr(self._Y @ self._x - self._tau) / self._num_samples
        self._objective = error
        
        # Create an NLP solver object
        nlp_prob = {
            'f': self._objective,
            'x': self._x,
            'g': self._constraints,
            'p': self._parameters
        }
        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }
        
        self._solver = ca.nlpsol('solver', 'ipopt', nlp_prob)
        
        # Solve the problem
        solution = self._solver(x0=np.zeros(self._nx), lbg=-ca.inf, ubg=ca.inf, max_iter=max_iter)
        
        return solution['x'].full().flatten()