import time
import numpy as np
import casadi as ca
from scipy.linalg import cholesky


class NonlinearLeastSquares():
    """
    Nonlinear least-squares inertial identification with:
      - log-Cholesky inertial parameterization: phi = phi(theta)
      - exponential friction parameterization:  b = exp(eta) (strictly positive)
      - pull-back regularization (constant metric at prior): 0.5*gamma*(phi-phi0)^T M (phi-phi0)

    Solved by Gauss-Newton / Levenberg-Marquardt on x = [theta; eta], using ONLY
    small precomputed normal-equation matrices (Y^T Y, Y^T tau, etc.)
    We don't construct big residual vectors of length 18*N (N=number of samples) or Jacobians of size (18*N x 130).
    """
    def __init__(
        self,
        regressor: np.ndarray,
        tau_vec: np.ndarray,
        num_links: int,
        phi_nominal: np.ndarray,
        B_v: np.ndarray = None,
        B_c: np.ndarray = None,
        exp_clip: float = 1e12, # prevents overflow in exp()
    ):
        # Dimensions
        self.num_links = int(num_links)
        self.num_inertial_params = 10
        self.nx_phi = self.num_links * self.num_inertial_params

        Y = np.asarray(regressor, dtype=np.float64)
        tau = np.asarray(tau_vec, dtype=np.float64).reshape(-1)
        assert Y.shape[1] == self.nx_phi, f"Y has {Y.shape[1]} cols, expected {self.nx_phi}"
        assert Y.shape[0] == tau.shape[0], "Y rows must match tau length"
        assert phi_nominal.shape[0] == self.nx_phi, "phi_nominal must be 10*num_links"

        self.Y = np.ascontiguousarray(Y)
        self.tau = np.ascontiguousarray(tau)
        self.phi_prior = np.ascontiguousarray(np.asarray(phi_nominal, dtype=np.float64).copy())

        self.exp_clip = float(exp_clip)

        # Precompute (Y^T @ Y), (Y^T @ tau), and (tau^T tau) for efficient cost and gradient computations.
        self.G = self.Y.T @ self.Y     # (10Lx10L)
        self.YTtau = self.Y.T @ self.tau # (10L,)
        self.tautau = float(self.tau @ self.tau)

        # Friction blocks
        self.identify_fric = (B_v is not None) and (B_c is not None)
        if self.identify_fric:
            Bv = np.ascontiguousarray(np.asarray(B_v, dtype=np.float64))
            Bc = np.ascontiguousarray(np.asarray(B_c, dtype=np.float64))
            assert Bv.shape[0] == self.Y.shape[0]
            assert Bc.shape[0] == self.Y.shape[0]
            self.num_joints = int(Bv.shape[1])
            assert Bc.shape[1] == self.num_joints

            self.B_v = Bv
            self.B_c = Bc

            # A = [Bv Bc] in block form
            self.YTBv = self.Y.T @ self.B_v            # Y^T @ Bv (10L x N_j)
            self.YTBc = self.Y.T @ self.B_c            # Y^T @ Bc (10L x N_j)
            self.K = np.hstack([self.YTBv, self.YTBc]) # Y^T @ A  (10L x 2N_j)

            # H = A^T A in block form
            H11 = self.B_v.T @ self.B_v
            H22 = self.B_c.T @ self.B_c
            H12 = self.B_v.T @ self.B_c
            self.H = np.block([[H11, H12],
                               [H12.T, H22]])  # (2N_j x 2N_j)

            # d = A^T tau
            dv = self.B_v.T @ self.tau
            dc = self.B_c.T @ self.tau
            self.d = np.hstack([dv, dc]) # (2N_j,)
        else:
            self.num_joints = 0
            self.B_v = self.B_c = None
            self.K = None
            self.H = None
            self.d = None

        # Pull-back metric at prior (constant)
        self.M_link = []
        for i in range(self.num_links):
            sl = slice(10 * i, 10 * (i + 1))
            self.M_link.append(self._pullback_metric(self.phi_prior[sl]))

        # Initial theta from nominal phi
        self.theta0 = self._phi_all_to_theta_all(self.phi_prior)

        # Build CasADi link maps phi(theta_link), dphi/dtheta_link
        self._build_casadi_link_maps()

    # -------------------- Inertial geometry -------------------- #
    def _construct_pseudo_inertia_matrix(self, phi: np.ndarray) -> np.ndarray:
        m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz = phi
        h = np.array([hx, hy, hz])
        I_bar = np.array([[Ixx, Ixy, Ixz],
                          [Ixy, Iyy, Iyz],
                          [Ixz, Iyz, Izz]], dtype=np.float64)
        
        J = np.zeros((4, 4), dtype=np.float64)
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

        # Extract theta parameters from U
        alpha = np.log(U[3, 3])
        U = U / U[3, 3]  # normalize so U[3,3]=1
        d1, d2, d3 = np.log(U[0, 0]), np.log(U[1, 1]), np.log(U[2, 2])
        s12, s13, s23 = U[0, 1], U[0, 2], U[1, 2]
        t1, t2, t3 = U[0, 3], U[1, 3], U[2, 3]
        theta = np.array([alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3], dtype=float)
        return theta

    def _phi_all_to_theta_all(self, phi_all: np.ndarray) -> np.ndarray:
        theta_all = np.zeros(self.nx_phi, dtype=np.float64)
        for i in range(self.num_links):
            sl = slice(10 * i, 10 * (i + 1))
            theta_all[sl] = self._phi_to_log_cholesky(phi_all[sl])
        return theta_all

    def _pullback_metric(self, phi_prior_link: np.ndarray) -> np.ndarray:
        """
        Constant pull-back metric M (10x10) at a numeric prior for the link.
        """
        M = np.zeros((self.num_inertial_params, self.num_inertial_params), dtype=np.float64)
        P = self._construct_pseudo_inertia_matrix(phi_prior_link)
        P_inv = np.linalg.solve(P, np.eye(4))

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
        M = 0.5 * (M + M.T)
        # Regularize M to ensure positive definiteness
        w = np.linalg.eigvalsh(M)
        if np.min(w) <= 1e-12:
            M = M + (1e-9 - np.min(w)) * np.eye(10)
        return M

    # -------------------- CasADi link maps -------------------- #
    def _build_casadi_link_maps(self):
        """
        Build CasADi functions for phi(theta) and its Jacobian dphi/dtheta for a single link.
        """
        th = ca.SX.sym("th", 10)  # [alpha,d1,d2,d3,s12,s13,s23,t1,t2,t3]
        alpha,d1,d2,d3,s12,s13,s23,t1,t2,t3 = th[0],th[1],th[2],th[3],th[4],th[5],th[6],th[7],th[8],th[9]

        def safe_exp(z):
            lo = -self.exp_clip
            hi = self.exp_clip
            return ca.exp(ca.fmin(ca.fmax(z, lo), hi))

        U = ca.SX.zeros(4, 4)
        U[0, 0] = safe_exp(d1); U[1, 1] = safe_exp(d2); U[2, 2] = safe_exp(d3)
        U[0, 1] = s12;         U[0, 2] = s13;         U[1, 2] = s23
        U[0, 3] = t1;          U[1, 3] = t2;          U[2, 3] = t3
        U[3, 3] = 1.0
        U = safe_exp(alpha) * U

        J = U.T @ U
        m = J[3, 3]
        h = J[0:3, 3]

        tr = ca.trace(J[0:3, 0:3])
        Ibar = tr * ca.SX.eye(3) - J[0:3, 0:3]
        Ixx, Iyy, Izz = Ibar[0, 0], Ibar[1, 1], Ibar[2, 2]
        Ixy, Ixz, Iyz = Ibar[0, 1], Ibar[0, 2], Ibar[1, 2]

        phi = ca.vertcat(m, h[0], h[1], h[2], Ixx, Ixy, Iyy, Ixz, Iyz, Izz)  # (10,)
        Jphi = ca.jacobian(phi, th)  # (10x10)

        self._phi_fun = ca.Function("phi_fun", [th], [phi])
        self._dphi_fun = ca.Function("dphi_fun", [th], [Jphi])

    def _phi_and_Jblocks(self, theta_all: np.ndarray):
        """Compute phi_all and J_blocks for all links given theta_all."""
        phi_all = np.zeros(self.nx_phi, dtype=np.float64)
        J_blocks = []
        for i in range(self.num_links):
            sl = slice(10 * i, 10 * (i + 1))
            th_i = theta_all[sl]
            phi_i = np.array(self._phi_fun(th_i)).reshape(-1)
            Ji = np.array(self._dphi_fun(th_i), dtype=np.float64)  # 10x10
            phi_all[sl] = phi_i
            J_blocks.append(Ji)
        return phi_all, J_blocks

    # -------------------- Helpers for exp friction GN -------------------- #
    def _exp_safe(self, eta: np.ndarray) -> np.ndarray:
        """Elementwise exp with clipping to avoid overflow."""
        return np.exp(np.clip(eta, -self.exp_clip, self.exp_clip))

    def _cost(self, phi: np.ndarray, b: np.ndarray, gamma: float) -> float:
        """
        Cost = 0.5||Y @ phi + A @ b - tau||^2 + 0.5*gamma*(phi-phi0)^T @ M @ (phi-phi0)
        computed via small matrices (no big residual vector).
        """
        # LS part: 0.5*(phi^T YTY phi + 2 phi^T YTA b + b^T H b - 2 phi^T YTtau - 2 b^T d + tau^T tau)
        cst = self.tautau
        val = phi @ (self.G @ phi) - 2.0 * (phi @ self.YTtau) + cst

        if self.identify_fric:
            val += 2.0 * (phi @ (self.K @ b))
            val += b @ (self.H @ b) - 2.0 * (b @ self.d)

        val *= 0.5

        # Regularization part: 0.5*gamma*(phi-phi0)^T @ M @ (phi-phi0)
        if gamma > 0.0:
            reg = 0.0
            for i in range(self.num_links):
                sl = slice(10 * i, 10 * (i + 1))
                dphi = phi[sl] - self.phi_prior[sl]
                reg += 0.5 * gamma * (dphi @ (self.M_link[i] @ dphi))
            val += reg

        return float(val)

    # -------------------- Main solver -------------------- #
    def solve_gn_exp(
        self,
        lambda_reg: float = 1e-4,
        max_iters: int = 100,
        tol: float = 1e-6,
        verbose: int = 1,
        damp0: float = 1e-2,
        max_ls: int = 12,
        max_damp_tries: int = 10,
    ):
        """
        Gauss-Newton / Levenberg-Marquardt on x=[theta; eta] with exp friction:
          b = exp(eta) (strictly positive)

        Uses only small precomputed matrices:
          G=Y^T Y, c=Y^T tau, K=Y^T A, H=A^T A, d=A^T tau
        and CasADi link Jacobians dphi/dtheta.

        Returns: (phi_hat, b_v, b_c, theta_hat, eta_hat)
        """
        gamma = float(lambda_reg)

        # Init
        theta = self.theta0.copy()

        if self.identify_fric:
            # Start with small positive friction => eta negative
            eta = -6.0 * np.ones(2 * self.num_joints, dtype=np.float64)
        else:
            eta = None

        # Initialize cost
        phi, J_blocks = self._phi_and_Jblocks(theta)
        if self.identify_fric:
            b = self._exp_safe(eta)
        else:
            b = None
        cost_prev = self._cost(phi, b, gamma)

        damp = float(damp0)
        t0 = time.time()

        for it in range(max_iters):
            # Forward
            phi, J_blocks = self._phi_and_Jblocks(theta)

            if self.identify_fric:
                b = self._exp_safe(eta)
                D = b  # Since diag(b) times a vector is elementwise multiply by b
            else:
                b = None
                D = None

            # ----- Gradients in phi- and b-space (small) -----
            # g_phi_ls = Y^T r = G phi + K b - c
            g_phi = self.G @ phi - self.YTtau
            if self.identify_fric:
                g_phi += self.K @ b

            # Add pull-back gradient in phi space: gamma M (phi-phi0)
            if gamma > 0.0:
                for i in range(self.num_links):
                    sl = slice(10 * i, 10 * (i + 1))
                    dphi = phi[sl] - self.phi_prior[sl]
                    g_phi[sl] += gamma * (self.M_link[i] @ dphi)

            # g_b_ls = A^T r = K^T phi + H b - d
            if self.identify_fric:
                g_b = (self.K.T @ phi) + (self.H @ b) - self.d
                # Chain to eta: g_eta = diag(b) * g_b
                g_eta = D * g_b
            else:
                g_b = None
                g_eta = None

            # ----- Chain to theta: g_theta = Jphi^T g_phi -----
            g_theta = np.zeros_like(theta)
            for i in range(self.num_links):
                sl = slice(10 * i, 10 * (i + 1))
                Ji = J_blocks[i]
                g_theta[sl] = Ji.T @ g_phi[sl]

            # ----- Hessian blocks (GN) -----
            # H_tt = Jphi^T G Jphi + Jphi^T (gamma M) Jphi
            H_tt = np.zeros((self.nx_phi, self.nx_phi), dtype=np.float64)

            # Dynamic part blockwise:
            for i in range(self.num_links):
                si = slice(10 * i, 10 * (i + 1))
                Ji = J_blocks[i]
                for j in range(self.num_links):
                    sj = slice(10 * j, 10 * (j + 1))
                    Gblk = self.G[si, sj]  # 10x10
                    H_tt[si, sj] += Ji.T @ Gblk @ J_blocks[j]

            # Regularization part: only diagonal blocks
            if gamma > 0.0:
                for i in range(self.num_links):
                    si = slice(10 * i, 10 * (i + 1))
                    Ji = J_blocks[i]
                    H_tt[si, si] += Ji.T @ (gamma * self.M_link[i]) @ Ji

            if self.identify_fric:
                # H_ee = (diag(b)) H (diag(b))
                H_ee = (D[:, None] * self.H) * D[None, :]

                # H_te = Jphi^T K diag(b)
                # Build blockwise: for link i, H_te[si,:] = Ji^T * (K[si,:] * b)
                H_te = np.zeros((self.nx_phi, 2 * self.num_joints), dtype=np.float64)
                KDb = self.K * D[None, :]  # 130x24 (column-scaled)
                for i in range(self.num_links):
                    si = slice(10 * i, 10 * (i + 1))
                    Ji = J_blocks[i]
                    H_te[si, :] = Ji.T @ KDb[si, :]

                H_et = H_te.T
            else:
                H_ee = H_te = H_et = None

            # ----- Solve damped system + backtracking -----
            accepted = False
            dtheta = None
            deta = None
            theta_new = None
            eta_new = None
            cost_new = None

            for _try in range(max_damp_tries):
                if self.identify_fric:
                    # Assemble full system
                    nT = self.nx_phi
                    nE = 2 * self.num_joints
                    H = np.block([
                        [H_tt + damp * np.eye(nT), H_te                    ],
                        [H_et,                     H_ee + damp * np.eye(nE)]
                    ])
                    g = np.hstack([g_theta, g_eta])
                else:
                    H = H_tt + damp * np.eye(self.nx_phi)
                    g = g_theta

                try:
                    dx = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    damp *= 10.0
                    continue

                if self.identify_fric:
                    dtheta = dx[:self.nx_phi]
                    deta = dx[self.nx_phi:]
                else:
                    dtheta = dx
                    deta = None

                step = 1.0
                for _ls in range(max_ls):
                    th_try = theta + step * dtheta
                    phi_try, _ = self._phi_and_Jblocks(th_try)

                    if self.identify_fric:
                        et_try = eta + step * deta
                        b_try = self._exp_safe(et_try)
                    else:
                        et_try = None
                        b_try = None

                    c_try = self._cost(phi_try, b_try, gamma)

                    if np.isfinite(c_try) and c_try < cost_prev:
                        accepted = True
                        theta_new = th_try
                        eta_new = et_try
                        cost_new = c_try
                        break

                    step *= 0.5

                if accepted:
                    break

                damp *= 10.0

            if not accepted:
                if verbose:
                    print(f"[GN-exp] it={it:02d} failed to find descent step; stopping.")
                break

            rel_impr = abs(cost_prev - cost_new) / (abs(cost_prev) + 1e-12)
            theta = theta_new
            if self.identify_fric:
                eta = eta_new
            cost_prev = cost_new

            damp = max(damp / 3.0, 1e-12)

            if verbose:
                if self.identify_fric:
                    b_now = self._exp_safe(eta)
                    bmin, bmax = float(np.min(b_now)), float(np.max(b_now))
                    print(f"[GN-exp] it={it:02d} cost={cost_prev:.6e} step={step:.2e} damp={damp:.2e} |dtheta|={np.linalg.norm(dtheta):.2e} b: [{bmin:.2e},{bmax:.2e}]")
                else:
                    print(f"[GN-exp] it={it:02d} cost={cost_prev:.6e} step={step:.2e} damp={damp:.2e} |dtheta|={np.linalg.norm(dtheta):.2e}")

            if rel_impr < tol:
                break

        dt = time.time() - t0

        # Final params
        phi_hat, _ = self._phi_and_Jblocks(theta)
        if self.identify_fric:
            b_hat = self._exp_safe(eta)
            b_v = b_hat[:self.num_joints]
            b_c = b_hat[self.num_joints:]
        else:
            b_v = b_c = None
        if verbose:
                print("\n", "-"*20, "Summary", "-"*20)
                print(f"[GN-exp] done in {dt:.2f}s, final cost={cost_prev:.6e}")
        return phi_hat, b_v, b_c