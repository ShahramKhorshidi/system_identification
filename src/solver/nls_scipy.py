import time
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import least_squares


class NonlinearLeastSquares():
    """
    Offline least-squares optimization to identify the inertial parameters of a chain of rigid bodies.
    It uses log-Cholesky parameterization to enforce physical feasibility.
    """
    def __init__(
            self,
            regressor: np.ndarray, 
            tau_vec: np.ndarray,
            num_links: int,
            phi_nominal: np.ndarray,
            B_v: np.ndarray = None,
            B_c: np.ndarray = None,
        ):
        self.Y = np.asarray(regressor)
        self.tau = np.asarray(tau_vec).reshape(-1)
        self.Y   = np.ascontiguousarray(self.Y, dtype=np.float64)
        self.tau = np.ascontiguousarray(self.tau, dtype=np.float64)

        self.num_links = int(num_links)
        self.num_inertial_params = 10
        self.nx_phi = self.num_links * self.num_inertial_params

        assert self.Y.shape[1] == self.nx_phi, f"Y has {self.Y.shape[1]} cols, expected {self.nx_phi}"
        assert self.Y.shape[0] == self.tau.shape[0], "Y rows must match tau length"
        assert phi_nominal.shape[0] == self.nx_phi, "phi_nominal must be 10*num_links"

        self.phi_prior = np.asarray(phi_nominal).copy()

        self.identify_fric = (B_v is not None) and (B_c is not None)
        if self.identify_fric:
            assert B_v.shape[0] == self.Y.shape[0]
            assert B_c.shape[0] == self.Y.shape[0]
            self.B_v = np.asarray(B_v)  # Viscous friction coefficient (Nm / (rad/s))
            self.B_c = np.asarray(B_c)  # Coulomb friction coefficient (Nm)
            self.num_joints = self.B_v.shape[1]
            self.theta_b0 = np.zeros(self.num_joints)
            self.theta_c0 = np.zeros(self.num_joints)
        else:
            self.B_v = None
            self.B_c = None
            self.num_joints = 0

        # Precompute constant pull-back metric per-link at prior (10x10), and its Cholesky factor
        self.M_link = []
        self.L_link = []
        for i in range(self.num_links):
            sl = slice(10 * i, 10 * (i + 1))
            M = self._pullback_metric(self.phi_prior[sl])
            L = cholesky(M, lower=True)  # M = L @ L.T
            self.M_link.append(M)
            self.L_link.append(L)

        # Initial theta from nominal phi
        self.theta0 = self._phi_all_to_theta_all(self.phi_prior)

    # -------------------- Geometry & Parameterization -------------------- #   
    def _construct_pseudo_inertia_matrix(self, phi: np.ndarray) -> np.ndarray:
        # Reconstruct pseudo inertia matrix J from standard parameters phi
        m, h_x, h_y, h_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz = phi
        h = np.array([h_x, h_y, h_z])
        I_bar = np.array([[Ixx, Ixy, Ixz],
                          [Ixy, Iyy, Iyz],
                          [Ixz, Iyz, Izz]])
        J = np.zeros((4, 4))
        J[0:3, 0:3] = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
        J[0:3, 3] = h
        J[3, 0:3] = h.T
        J[3, 3] = m
        return J

    def _phi_to_log_cholesky(self, phi: np.ndarray) -> np.ndarray:
        #Convert standard inertial parameters phi to log-Cholesky theta.
        J = self._construct_pseudo_inertia_matrix(phi)
        
        # Compute upper-triangular Cholesky decomposition
        U = cholesky(J, lower=False)
        
        # Extract theta parameters
        alpha = np.log(U[3, 3])
        U /= U[3, 3] # Normalize U by exp(alpha) to ensure U[3, 3] = 1
        d1, d2, d3 = np.log(U[0,0]), np.log(U[1,1]), np.log(U[2,2])
        s12, s13, s23 = U[0,1], U[0,2], U[1,2]
        t1, t2, t3 = U[0,3], U[1,3], U[2,3]
        theta = np.array([alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3], dtype=float)
        return theta

    def _log_cholesky_to_phi(self, theta_link: np.ndarray) -> np.ndarray:
        """
        Maps theta to standard inertial parameters phi using J = U^T @ U,
        where U is the upper-triangular Cholesky factor parameterized by theta.
        Ref:"Wensing, Smooth Parameterization of Rigid-Body Inertia"
        """
        alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3 = theta_link
        
        # Parametric equations for the inertial parameters, phi = f(theta)
        # m = (t1**2 + t2**2 + t3**2+1)
        # h1 = (t1 * np.exp(d1))
        # h2 = (t1 * s12 + t2 * np.exp(d2))
        # h3 = (t1 * s13 + t2 * s23 + t3 * np.exp(d3))
        # Ixx = (s12**2 + s13**2 + s23**2 + np.exp(2*d2) + np.exp(2*d3))
        # Iyy = (s13**2 + s23 ** 2 + np.exp(2 * d1) + np.exp(2 * d3))
        # Izz = (s12**2 + np.exp(2*d1) + np.exp(2*d2))
        # Ixy = (-s12 * np.exp(d1))
        # Iyz =  (-s12 * s13 - s23 * np.exp(d2))
        # Ixz = (-s13 * np.exp(d1))
        # phi_link = np.exp(2*alpha) * np.array([m, h1, h2, h3, Ixx, Ixy, Iyy, Ixz, Iyz, Izz], dtype=float)
        
        # Using the log-Cholesky parameterization
        U = np.zeros((4,4))
        U[0,0] = np.exp(d1); U[1,1] = np.exp(d2); U[2,2] = np.exp(d3)
        U[0,1] = s12; U[0,2] = s13; U[1,2] = s23
        U[0,3] = t1;  U[1,3] = t2;  U[2,3] = t3
        U[3,3] = 1.0
        U *= np.exp(alpha)
        J = U.T @ U
        m = J[3, 3]
        h = J[:3, 3]
        I_bar = np.trace(J[:3, :3]) * np.eye(3) - J[:3, :3]
        Ixx, Iyy, Izz = I_bar[0,0], I_bar[1,1], I_bar[2,2]
        Ixy, Ixz, Iyz = I_bar[0,1], I_bar[0,2], I_bar[1,2]
        phi = np.hstack((m, h, Ixx, Ixy, Iyy, Ixz, Iyz, Izz))
        return phi
    
    def _pullback_metric(self, phi):
        # Returns the approximation of Riemannian distance metric (M:10x10) as a numpy array
        M = np.zeros((self.num_inertial_params, self.num_inertial_params))
        P = self._construct_pseudo_inertia_matrix(phi)
        P_inv = np.linalg.inv(P)
        
        for i in range(10):
            v_i = np.zeros(10)
            v_i[i] = 1
            V_i = self._construct_pseudo_inertia_matrix(v_i)
            for j in range(10):
                v_j = np.zeros(10)
                v_j[j] = 1
                V_j = self._construct_pseudo_inertia_matrix(v_j)
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
        
     # -------------------- Vectorized over links -------------------- #
    def _phi_all_to_theta_all(self, phi_all: np.ndarray) -> np.ndarray:
        theta_all = np.zeros(self.nx_phi)
        for i in range(self.num_links):
            sl = slice(10*i, 10*(i+1))
            theta_all[sl] = self._phi_to_log_cholesky(phi_all[sl])
        return theta_all

    def _theta_all_to_phi_all(self, theta_all: np.ndarray) -> np.ndarray:
        phi_all = np.zeros(self.nx_phi)
        for i in range(self.num_links):
            sl = slice(10*i, 10*(i+1))
            phi_all[sl] = self._log_cholesky_to_phi(theta_all[sl])
        return phi_all
    
    # -------------------- Residual for least_squares -------------------- #
    def _residual(self, x: np.ndarray) -> np.ndarray:
        """
        x = [theta_all (10L), theta_b (nJ), theta_c (nJ)] if friction enabled.
        Friction is mapped as: b = exp(theta_b), bc = exp(theta_c), for positivity.
        """
        theta_all = x[:self.nx_phi]
        phi_all = self._theta_all_to_phi_all(theta_all)
        phi_all = np.ascontiguousarray(phi_all, dtype=np.float64)

        if self.identify_fric:
            off = self.nx_phi
            theta_b = x[off:off + self.num_joints]
            theta_c = x[off + self.num_joints:off + 2 * self.num_joints]
            b_v = np.exp(theta_b)
            b_c = np.exp(theta_c)
            dyn = self.Y.dot(phi_all) + self.B_v.dot(b_v) + self.B_c.dot(b_c) - self.tau
        else:
            dyn = self.Y.dot(phi_all) - self.tau

        # pull-back regularization (constant metric at prior)
        # Add residual blocks per-link: sqrt(gamma) * L_prior * (phi_link - phi_prior_link)
        reg_blocks = []
        if self.gamma > 0.0:
            sqrt_gamma = np.sqrt(self.gamma)
            for i in range(self.num_links):
                sl = slice(10 * i, 10 * (i + 1))
                dphi = (phi_all[sl] - self.phi_prior[sl])
                # L is lower-tri such that M = L L^T
                reg_blocks.append(sqrt_gamma * (self.L_link[i] @ dphi))
        reg = np.concatenate(reg_blocks) if reg_blocks else np.zeros(0)

        return np.hstack([dyn, reg])

    # -------------------- API -------------------- #
    def solve_nls_batch(
            self,
            lambda_reg:float = 1e-4,
            max_nfev:int = 10000,
            verbose:int = 2
        ):
        self.gamma = lambda_reg
        if self.identify_fric:
            x0 = np.hstack([self.theta0, self.theta_b0, self.theta_c0])
            lb = np.full_like(x0, -np.inf)
            ub = np.full_like(x0,  np.inf)
        else:
            x0 = self.theta0.copy()
            lb, ub = -np.inf, np.inf

        t0 = time.time()
        res = least_squares(
            self._residual,
            x0,
            method="trf",
            jac="2-point",
            tr_solver="lsmr",
            x_scale="jac",
            bounds=(lb, ub),
            ftol=1e-5, xtol=1e-5, gtol=1e-5,
            max_nfev=max_nfev,
            verbose=verbose,
        )
        dt = time.time() - t0

        theta_hat = res.x[:self.nx_phi]
        phi_hat = self._theta_all_to_phi_all(theta_hat)

        if self.identify_fric:
            off = self.nx_phi
            theta_b = res.x[off:off + self.num_joints]
            theta_c = res.x[off + self.num_joints:off + 2 * self.num_joints]
            b_v = np.exp(theta_b)
            b_c = np.exp(theta_c)
        else:
            b_v = b_c = None
        print(f"[NLS-batch] success={res.success}, cost={0.5*np.sum(res.fun**2):.3e}, time={dt:.3f}s")
        return phi_hat, b_v, b_c