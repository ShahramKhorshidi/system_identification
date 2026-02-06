import time
import numpy as np
from collections import deque
from scipy.optimize import least_squares


class NonlinearLeastSquares():
    """
    Moving-window least-squares to estimate the inertial parameters of ONE added rigid body to the robot.
    It uses log-Cholesky parameterization to enforce physical feasibility.
    Model: tau_t = Y_nom_t @ phi_nom  +  Y_est_t @ phi_est + noise
    """
    def __init__(self,
                 n_dof: int,
                 phi_nom: np.ndarray,          # (10 * n_links_nominal,)
                 buffer_size: int = 50,        # horizon length
                 lambda_reg: float = 1e-3,     # strength on (phi_est - phi_prior)
                 rho_process: float = 0.0,     # strength on (theta - theta_prev) (static by default)
                 phi_prior_est: np.ndarray = None):  # 10-dim prior for the estimated inertia
        self.n_dof = n_dof
        self.phi_nom = phi_nom.copy()
        self.buffer_size = buffer_size
        self.lambda_reg = float(lambda_reg)
        self.rho_process = float(rho_process)
        
        # --- Prior for the added body (tiny, SPD, "almost zero")
        if phi_prior_est is None:
            self.phi_prior_est = self._small_prior()  # [m, h, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        else:
            self.phi_prior_est = phi_prior_est.astype(float)

        # --- Buffers (horizon)
        self.Y_nom_buf = deque(maxlen=buffer_size)  # each: (n_dof, 10*n_links_nominal)
        self.Y_est_buf = deque(maxlen=buffer_size)  # each: (n_dof, 10)
        self.tau_buf   = deque(maxlen=buffer_size)  # each: (n_dof,)

        # --- Current parameter in log-Cholesky space (theta is 10-dim)
        self.theta_prev = self.phi_to_log_cholesky(self.phi_prior_est)
        self.theta_hat  = self.theta_prev.copy()
        self.result     = None
        
    @staticmethod
    def _small_prior(m=0.05, I=5e-4):
        """A small but strictly physical prior (sphere-like inertia, zero first moments)."""
        # phi = [m, h1, h2, h3, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        return np.array([m, 0.0, 0.0, 0.0, I, 0.0, I, 0.0, 0.0, I], dtype=float)
    
    @staticmethod
    def construct_pseudo_inertia(phi_link: np.ndarray) -> np.ndarray:
        """Builds the 4x4 pseudo-inertia matrix J from standard 10 inertial params."""
        m = phi_link[0]
        h = phi_link[1:4]
        Ixx, Ixy, Iyy, Ixz, Iyz, Izz = phi_link[4:]
        I_bar = np.array([[Ixx, Ixy, Ixz],
                          [Ixy, Iyy, Iyz],
                          [Ixz, Iyz, Izz]], dtype=float)
        J = np.zeros((4, 4), dtype=float)
        J[:3, :3] = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
        J[:3,  3] = h
        J[ 3, :3] = h
        J[ 3,  3] = m
        return J

    def phi_to_log_cholesky(self, phi_link: np.ndarray) -> np.ndarray:
        #Convert standard inertial parameters phi to log-Cholesky theta.
        J = self.construct_pseudo_inertia(phi_link)

        # Compute upper-triangular Cholesky decomposition
        U = np.linalg.cholesky(J, upper=True)
        
        # Extract theta parameters
        alpha = np.log(U[3, 3])
        # Normalize U by exp(alpha) to ensure U[3, 3] = 1
        U /= np.exp(alpha)
        d1 = np.log(U[0, 0])
        s12 = U[0, 1]
        s13 = U[0, 2]
        t1 = U[0, 3]
        d2 = np.log(U[1, 1])
        s23 = U[1, 2]
        t2 = U[1, 3]
        d3 = np.log(U[2, 2])
        t3 = U[2, 3]

        theta_link = np.array([d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha], dtype=float)
        return theta_link

    def log_cholesky_to_phi(self, theta_link: np.ndarray, phi_prior_link: np.ndarray = None) -> np.ndarray:
        """
        Maps theta to standard inertial parameters phi using J = U^T J0 U,
        with U upper-tri built from theta and J0 from prior.
        Ref:"Wensing, Smooth Parameterization of Rigid-Body Inertia"
        """
        d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha = theta_link
        
        # Parametricetric equations for the inertial parameters, phi = f(theta)
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
        U = np.zeros((4, 4), dtype=float)
        U[0, 0] = np.exp(d1)
        U[0, 1] = s12
        U[0, 2] = s13
        U[0, 3] = t1
        U[1, 1] = np.exp(d2)
        U[1, 2] = s23
        U[1, 3] = t2
        U[2, 2] = np.exp(d3)
        U[2, 3] = t3
        U[3, 3] = 1.0
        U *= np.exp(alpha)
        
        # Compute the pseudo-inertia matrix from U
        if phi_prior_link is None:
            J = U.T @ np.eye(4) @ U
        else:
            J0 = self.construct_pseudo_inertia(phi_prior_link)
            J = U.T @ J0 @ U
        
        # Extract the inertial parameters from J
        m = J[3, 3]
        h = J[0:3, 3]
        I_bar = np.trace(J[0:3, 0:3]) * np.eye(3) - J[0:3, 0:3]
        Ixx, Iyy, Izz = I_bar[0,0], I_bar[1,1], I_bar[2,2]
        Ixy, Ixz, Iyz = I_bar[0,1], I_bar[0,2], I_bar[1,2]
        phi_link = np.array([m, h[0], h[1], h[2], Ixx, Ixy, Iyy, Ixz, Iyz, Izz], dtype=float)
        return phi_link
    
    # ---------- Data flow ---------- #
    def update(self, Y_nom_t: np.ndarray, Y_est_t: np.ndarray, tau_t: np.ndarray):
        """Push one sample into the horizon buffers."""
        assert Y_nom_t.shape[0] == self.n_dof and tau_t.shape[0] == self.n_dof
        assert Y_est_t.shape == (self.n_dof, 10)
        self.Y_nom_buf.append(Y_nom_t)
        self.Y_est_buf.append(Y_est_t)
        self.tau_buf.append(tau_t)

    # -------- Least-squares -------- #
    def _residual(self, x: np.ndarray) -> np.ndarray:
        theta_link = x[:10]
        phi_est = self.log_cholesky_to_phi(theta_link)

        dyn_res = []
        for k in range(len(self.tau_buf)):
            Y_nom = self.Y_nom_buf[k]
            Y_est = self.Y_est_buf[k]
            tau   = self.tau_buf[k]
            tau_pred = Y_nom @ self.phi_nom + Y_est @ phi_est
            dyn_res.append(tau_pred - tau)

        dyn_res = np.concatenate(dyn_res) if dyn_res else np.zeros(0)

        # Regularization: keep phi_add close to tiny prior (Riemannian-lite)
        reg_phi = np.sqrt(self.lambda_reg) * (phi_est - self.phi_prior_est)

        # Optional process regularization in theta-space (static by default -> rho=0)
        reg_proc = np.sqrt(self.rho_process) * (theta_link - self.theta_prev)

        return np.concatenate([dyn_res, reg_phi, reg_proc])

    def solve(self, max_nfev=100, verbose=1):
        if len(self.tau_buf) == 0:
            raise RuntimeError("No samples in the horizon. Call update(...) first.")

        x0 = self.theta_hat.copy()
        start = time.time()
        result = least_squares(self._residual,
                               x0,
                               method="trf",
                               ftol=5e-5, xtol=1e-6, gtol=1e-6,
                               max_nfev=max_nfev,
                               verbose=verbose)
        duration = time.time() - start

        if result.success:
            self.theta_hat = result.x[:10].copy()
            self.result = result
            self.theta_prev = self.theta_hat.copy()
            print(f"[OK] LS solved in {duration:.4f}s, cost={0.5*np.sum(result.fun**2):.3e}")
        else:
            print("[WARN] LS failed:", result.message)

    def get_est_body_phi(self) -> np.ndarray:
        """Returns the estimated added-body inertial parameters (10,)."""
        return self.log_cholesky_to_phi(self.theta_hat)