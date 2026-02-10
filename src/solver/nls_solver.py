import time
import numpy as np
import casadi as ca
from scipy.linalg import cholesky


class NonlinearLeastSquares():
    """
    Offline nonlinear-least-squares optimization to identify the inertial parameters of a chain of rigid bodies.
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
        self.num_links = int(num_links)
        self.num_inertial_params = 10
        self.nx_phi = self.num_links * self.num_inertial_params

        assert regressor.shape[1] == self.nx_phi, f"Y has {regressor.shape[1]} cols, expected {self.nx_phi}"
        assert regressor.shape[0] == tau_vec.shape[0], "Y rows must match tau length"
        assert phi_nominal.shape[0] == self.nx_phi, "phi_nominal must be 10*num_links"

        self.phi_prior = np.asarray(phi_nominal).copy()

        self.Y = np.asarray(regressor)
        self.tau = np.asarray(tau_vec).reshape(-1)
        self.Y   = np.ascontiguousarray(self.Y, dtype=np.float64)
        self.tau = np.ascontiguousarray(self.tau, dtype=np.float64)
        self.YTY   = self.Y.T @ self.Y          # (130,130)
        self.YTtau = self.Y.T @ self.tau        # (130,)
        self.tautau = float(self.tau @ self.tau)

        self.identify_fric = (B_v is not None) and (B_c is not None)
        if self.identify_fric:
            assert B_v.shape[0] == self.Y.shape[0]
            assert B_c.shape[0] == self.Y.shape[0]
            self.B_v = np.asarray(B_v)  # Viscous friction coefficient (Nm / (rad/s))
            self.B_c = np.asarray(B_c)  # Coulomb friction coefficient (Nm)
            self.num_joints = self.B_v.shape[1]
            self.B_v = np.ascontiguousarray(self.B_v, dtype=np.float64)
            self.B_c = np.ascontiguousarray(self.B_c, dtype=np.float64)

            self.YTBv = self.Y.T @ self.B_v     # (130x12)
            self.YTBc = self.Y.T @ self.B_c     # (130x12)
            self.BvTBv = self.B_v.T @ self.B_v  # (12x12)
            self.BcTBc = self.B_c.T @ self.B_c  # (12x12)
            self.BvTBc = self.B_v.T @ self.B_c  # (12x12)
            self.BvTtau = self.B_v.T @ self.tau # (12,)
            self.BcTtau = self.B_c.T @ self.tau # (12,)
        else:
            self.B_v = None
            self.B_c = None
            self.num_joints = 0

        # Precompute constant pull-back metric per-link at prior (10x10)
        self.M_link = []
        for i in range(self.num_links):
            sl = slice(10 * i, 10 * (i + 1))
            M = self._pullback_metric(self.phi_prior[sl])
            self.M_link.append(M)

        # Initial theta from nominal phi
        self.theta0 = self._phi_all_to_theta_all(self.phi_prior)

        self._build_casadi_link_maps()

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

    def _phi_all_to_theta_all(self, phi_all: np.ndarray) -> np.ndarray:
        theta_all = np.zeros(self.nx_phi)
        for i in range(self.num_links):
            sl = slice(10*i, 10*(i+1))
            theta_all[sl] = self._phi_to_log_cholesky(phi_all[sl])
        return theta_all

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

    def _build_casadi_link_maps(self):
        th = ca.SX.sym("th", 10)  # [alpha,d1,d2,d3,s12,s13,s23,t1,t2,t3]
        alpha,d1,d2,d3,s12,s13,s23,t1,t2,t3 = th[0],th[1],th[2],th[3],th[4],th[5],th[6],th[7],th[8],th[9]

        U = ca.SX.zeros(4,4)
        def safe_exp(z, lo=-40, hi=40):
            return ca.exp(ca.fmin(ca.fmax(z, lo), hi))
        U[0,0] = safe_exp(d1); U[1,1] = safe_exp(d2); U[2,2] = safe_exp(d3)
        U[0,1] = s12;        U[0,2] = s13;        U[1,2] = s23
        U[0,3] = t1;         U[1,3] = t2;         U[2,3] = t3
        U[3,3] = 1.0
        U = safe_exp(alpha) * U

        J = U.T @ U
        m = J[3,3]
        h = J[0:3, 3]

        tr = ca.trace(J[0:3,0:3])
        Ibar = tr * ca.SX.eye(3) - J[0:3,0:3]
        Ixx = Ibar[0,0]; Iyy = Ibar[1,1]; Izz = Ibar[2,2]
        Ixy = Ibar[0,1]; Ixz = Ibar[0,2]; Iyz = Ibar[1,2]

        phi = ca.vertcat(m, h[0], h[1], h[2], Ixx, Ixy, Iyy, Ixz, Iyz, Izz)  # (10,)

        Jphi = ca.jacobian(phi, th)  # (10x10)

        self._phi_fun = ca.Function("phi_fun", [th], [phi])
        self._dphi_fun = ca.Function("dphi_fun", [th], [Jphi])

    def _phi_and_block_jphi(self, theta_all: np.ndarray):
        phi_all = np.zeros(self.nx_phi)
        J_blocks = []  # list of 10x10

        for i in range(self.num_links):
            sl = slice(10*i, 10*(i+1))
            th_i = theta_all[sl]

            phi_i = np.array(self._phi_fun(th_i)).reshape(-1)
            J_i   = np.array(self._dphi_fun(th_i))  # 10x10

            phi_all[sl] = phi_i
            J_blocks.append(J_i)

        return phi_all, J_blocks

    # -------------------- API -------------------- #
    def solve_gn(
    self,
    lambda_reg: float = 1e-4,
    max_iters: int = 30,
    tol: float = 1e-6,
    verbose: int = 1,
    ):
        """
        Gauss–Newton / Levenberg–Marquardt in theta-space with:
        - CasADi exact per-link J_phi = dphi/dtheta
        - Pull-back regularization in phi-space (constant metric at prior)
        - Friction solved in b-space (NO exp) via coupled 24x24 normal equations + positivity clamp
        - Backtracking line-search + adaptive damping to prevent NaNs/overflow
        """
        gamma = float(lambda_reg)
        self.gamma = gamma

        # -------- init --------
        theta = self.theta0.copy()

        # friction in b-space (not log-space!)
        if self.identify_fric:
            b_v = 1e-3 * np.ones(self.num_joints)
            b_c = 1e-3 * np.ones(self.num_joints)
        else:
            b_v = b_c = None

        # helper: cost without forming big residual
        def cost_from(phi, b_v=None, b_c=None):
            c = 0.5 * (phi @ (self.YTY @ phi) - 2.0 * (phi @ self.YTtau) + self.tautau)

            if self.identify_fric:
                c += (phi @ (self.YTBv @ b_v) + phi @ (self.YTBc @ b_c))
                c += 0.5 * (
                    b_v @ (self.BvTBv @ b_v)
                    + b_c @ (self.BcTBc @ b_c)
                    + 2.0 * b_v @ (self.BvTBc @ b_c)
                )
                c -= (b_v @ self.BvTtau + b_c @ self.BcTtau)

            if gamma > 0.0:
                reg = 0.0
                for i in range(self.num_links):
                    sl = slice(10 * i, 10 * (i + 1))
                    dphi = phi[sl] - self.phi_prior[sl]
                    reg += 0.5 * gamma * (dphi @ (self.M_link[i] @ dphi))
                c += reg

            return float(c)

        # initial phi and cost
        phi, J_blocks = self._phi_and_block_jphi(theta)
        cost_prev = cost_from(phi, b_v, b_c) if self.identify_fric else cost_from(phi)

        # Levenberg damping (adaptive)
        damp = 1e-2

        t0 = time.time()

        for it in range(max_iters):
            # --- forward ---
            phi, J_blocks = self._phi_and_block_jphi(theta)

            # --- friction optimal step in b-space (coupled 24x24) ---
            if self.identify_fric:
                # gradients in b-space
                g_bv = (self.YTBv.T @ phi) + (self.BvTBv @ b_v) + (self.BvTBc @ b_c) - self.BvTtau
                g_bc = (self.YTBc.T @ phi) + (self.BvTBc.T @ b_v) + (self.BcTBc @ b_c) - self.BcTtau

                H_b = np.block([
                    [self.BvTBv,     self.BvTBc],
                    [self.BvTBc.T,   self.BcTBc]
                ]) + 1e-9 * np.eye(2 * self.num_joints)

                db = -np.linalg.solve(H_b, np.hstack([g_bv, g_bc]))
                b_v_cand = b_v + db[:self.num_joints]
                b_c_cand = b_c + db[self.num_joints:]

                # enforce non-negativity (simple projection)
                b_v_cand = np.maximum(b_v_cand, 0.0)
                b_c_cand = np.maximum(b_c_cand, 0.0)
            else:
                b_v_cand = b_c_cand = None

            # --- gradient in phi-space: g_phi = Y^T r + reg_grad ---
            g_phi = self.YTY @ phi - self.YTtau
            if self.identify_fric:
                g_phi += self.YTBv @ b_v_cand + self.YTBc @ b_c_cand

            # pull-back reg gradient in phi-space (blockwise)
            if gamma > 0.0:
                for i in range(self.num_links):
                    sl = slice(10 * i, 10 * (i + 1))
                    dphi = phi[sl] - self.phi_prior[sl]
                    g_phi[sl] += gamma * (self.M_link[i] @ dphi)

            # --- chain rule to theta-space: g_theta = Jphi^T g_phi ---
            g_theta = np.zeros_like(theta)
            for i in range(self.num_links):
                sl = slice(10 * i, 10 * (i + 1))
                Ji = J_blocks[i]  # 10x10
                g_theta[sl] = Ji.T @ g_phi[sl]

            # --- GN Hessian in theta-space: H ≈ Jphi^T (Y^T Y) Jphi + reg ---
            H_theta = np.zeros((self.nx_phi, self.nx_phi))

            # dynamic part (blockwise 10x10 blocks)
            for i in range(self.num_links):
                si = slice(10 * i, 10 * (i + 1))
                Ji = J_blocks[i]
                for j in range(self.num_links):
                    sj = slice(10 * j, 10 * (j + 1))
                    Yblk = self.YTY[si, sj]
                    H_theta[si, sj] += Ji.T @ Yblk @ J_blocks[j]

            # reg part (diagonal blocks only)
            if gamma > 0.0:
                for i in range(self.num_links):
                    si = slice(10 * i, 10 * (i + 1))
                    Ji = J_blocks[i]
                    H_theta[si, si] += Ji.T @ (gamma * self.M_link[i]) @ Ji

            # --- solve for step with adaptive damping + backtracking ---
            accepted = False
            dtheta = None
            theta_new = None
            cost_new = None

            for attempt in range(10):
                H_damped = H_theta + damp * np.eye(self.nx_phi)

                try:
                    dtheta = -np.linalg.solve(H_damped, g_theta)
                except np.linalg.LinAlgError:
                    damp *= 10.0
                    continue

                step = 1.0
                for _ls in range(12):
                    theta_try = theta + step * dtheta
                    phi_try, _ = self._phi_and_block_jphi(theta_try)

                    if self.identify_fric:
                        c_try = cost_from(phi_try, b_v_cand, b_c_cand)
                    else:
                        c_try = cost_from(phi_try)

                    if np.isfinite(c_try) and c_try < cost_prev:
                        accepted = True
                        theta_new = theta_try
                        cost_new = c_try
                        break

                    step *= 0.5

                if accepted:
                    break

                # no descent with this damping -> increase and retry
                damp *= 10.0

            if not accepted:
                if verbose:
                    print(f"[GN] it={it:02d} failed to find descent step; stopping.")
                break

            # accept update
            theta = theta_new
            if self.identify_fric:
                b_v, b_c = b_v_cand, b_c_cand

            rel_impr = abs(cost_prev - cost_new) / (abs(cost_prev) + 1e-12)
            cost_prev = cost_new

            # decrease damping after success
            damp = max(damp / 3.0, 1e-12)

            if verbose:
                print(f"[GN] it={it:02d} cost={cost_prev:.6e} step={step:.3e} damp={damp:.3e} |dtheta|={np.linalg.norm(dtheta):.3e}")

            if rel_impr < tol:
                break

        dt = time.time() - t0

        # final phi
        phi_hat, _ = self._phi_and_block_jphi(theta)

        if self.identify_fric:
            print(f"[GN] done in {dt:.2f}s, final cost={cost_prev:.6e}")
            return phi_hat, b_v, b_c
        else:
            print(f"[GN] done in {dt:.2f}s, final cost={cost_prev:.6e}")
            return phi_hat, None, None
