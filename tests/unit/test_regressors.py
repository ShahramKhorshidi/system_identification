import os
import time
import numpy as np
import pinocchio as pin
from src.dynamics.quadrupd_dynamics import QuadrupedDynamics

def load_data(path, robot_name):
    path = os.path.join(path, "data/solo/")
    robot_q = np.loadtxt(path + robot_name + "_robot_q.dat", delimiter="\t", dtype=np.float64)
    robot_dq = np.loadtxt(path + robot_name + "_robot_dq.dat", delimiter="\t", dtype=np.float64)
    robot_ddq = np.loadtxt(path + robot_name + "_robot_ddq.dat", delimiter="\t", dtype=np.float64)
    robot_tau = np.loadtxt(path + robot_name + "_robot_tau.dat", delimiter="\t", dtype=np.float64)
    robot_contact = np.loadtxt(path + robot_name + "_robot_contact.dat", delimiter="\t", dtype=np.float64)
    robot_ee_force = np.loadtxt(path + robot_name + "_robot_ee_force.dat", delimiter="\t", dtype=np.float64)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact, robot_ee_force

def check_classical_regressor(quad_dyn, q, dq, ddq, tau, cnt, force, i):
    phi = quad_dyn.get_phi_nominal()
    quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])
    Y_classical = quad_dyn.get_classical_regressor(q[:, i], dq[:, i], ddq[:, i])
    tau_rnea = Y_classical @ phi
    tau_ext = (quad_dyn.S.T @ tau[:, i] + quad_dyn.get_cnt_force(force[:, i], cnt[:, i]))
    print("RNEA check ||tau_rnea - tau||:", np.linalg.norm(tau_rnea - tau_ext))
    print("rel:", np.linalg.norm(tau_rnea - tau_ext)/max(1e-12, np.linalg.norm(tau_ext)))

def check_W2_regressor(quad_dyn, q, dq, i):
    phi = quad_dyn.get_phi_nominal()
    quad_dyn.update_fk(q[:, i], dq[:, i], np.zeros(quad_dyn.nv))
    W1, W2 = quad_dyn.get_momentum_regressors(q[:, i], dq[:, i])
    H = pin.crba(quad_dyn.rmodel, quad_dyn.rdata, q[:, i])
    H = 0.5*(H+H.T)
    p_true = H @ dq[:, i]
    p_reg  = W2 @ phi
    print("W2 check ||p_true - p_reg||:", np.linalg.norm(p_true - p_reg))
    print("rel:", np.linalg.norm(p_true - p_reg)/max(1e-12, np.linalg.norm(p_true)))

def check_W1_regressor(quad_dyn, q, dq, i):
    phi = quad_dyn.get_phi_nominal()
    quad_dyn.update_fk(q[:, i], dq[:, i], np.zeros(quad_dyn.nv))
    W1, W2 = quad_dyn.get_momentum_regressors(q[:, i], dq[:, i])
    C = pin.computeCoriolisMatrix(quad_dyn.rmodel, quad_dyn.rdata, q[:, i], dq[:, i])
    g = pin.computeGeneralizedGravity(quad_dyn.rmodel, quad_dyn.rdata, q[:, i])
    d_true = C.T @ dq[:, i] - g
    d_reg = W1 @ phi
    print("W1 check ||d_true - d_reg||:", np.linalg.norm(d_true - d_reg))
    print("rel:", np.linalg.norm(d_true - d_reg)/max(1e-12, np.linalg.norm(d_true)))

def check_W1_with_dp(quad_dyn, q, dq, ddq, tau, cnt, force, dt, i):
    phi = quad_dyn.get_phi_nominal()
    quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])

    W1_k, W2_k = quad_dyn.get_momentum_regressors(q[:, i], dq[:, i])
    W1_k1, W2_k1 = quad_dyn.get_momentum_regressors(q[:, i+1], dq[:, i+1])
    f_ext_1 = quad_dyn.get_cnt_force(force[:, i], cnt[:, i])
    f_ext_2 = quad_dyn.get_cnt_force(force[:, i+1], cnt[:, i+1])
    
    p_k = W2_k @ phi
    p_k1 = W2_k1 @ phi

    lhs = (p_k1 - p_k) / dt
    rhs = 0.5*(quad_dyn.S.T@tau[:, i]+quad_dyn.S.T @ tau[:, i+1]) + 0.5*(W1_k@phi + W1_k1@phi)+ 0.5*(f_ext_1 + f_ext_2)
    r = lhs - rhs

    print("||lhs||", np.linalg.norm(lhs))
    print("||rhs||", np.linalg.norm(rhs))
    print("||r||  ", np.linalg.norm(r))
    print("base ||r||", np.linalg.norm(r[:6]))
    print("jnts ||r||", np.linalg.norm(r[6:]))
    
def check_momentum_vs_classical_window(quad_dyn, q, dq, ddq, tau, cnt, force, dt, i, h, theta, joint_only=False):
    nv = quad_dyn.nv
    # --- ensure contact schedule constant (otherwise P won't eliminate contact forces) ---
    # ensure cnt is constant during th ewhole window h
    for k in range(h):
        if np.any(cnt[:, i+k] != cnt[:, i]):
            print("[warn] contact schedule changes inside window; results may be invalid.")

    # --- Build P once for this window ---
    quad_dyn.update_fk(q[:, i], dq[:, i], np.zeros(nv))
    P = quad_dyn.get_null_space_proj(cnt[:, i])  # nv x nv

    # --- Build u = Sigma(P S^T tau dt) ---
    u = np.zeros(nv)
    for k in range(i, i+h):
        uk  = P @ (quad_dyn.S.T @ tau[:, k])
        uk1 = P @ (quad_dyn.S.T @ tau[:, k+1])
        u += 0.5 * (uk + uk1) * dt

    # --- Build integrated classical regressor Y_int = Sigma(P Y_L dt) ---
    Y_int = np.zeros((nv, theta.shape[0]))
    for k in range(i, i+h):
        quad_dyn.update_fk(q[:, k], dq[:, k], ddq[:, k])
        Yk = quad_dyn.get_classical_regressor(q[:, k], dq[:, k], ddq[:, k])
        quad_dyn.update_fk(q[:, k+1], dq[:, k+1], ddq[:, k+1])
        Yk1 = quad_dyn.get_classical_regressor(q[:, k+1], dq[:, k+1], ddq[:, k+1])
        Y_int += 0.5 * (P @ Yk + P @ Yk1) * dt

    # --- Build momentum window regressor Y_h ---
    _, W2_i  = quad_dyn.get_momentum_regressors(q[:, i],   dq[:, i])
    _, W2_ih = quad_dyn.get_momentum_regressors(q[:, i+h], dq[:, i+h])

    W1_int = np.zeros_like(W2_i)
    for k in range(i, i+h):
        W1_k, _   = quad_dyn.get_momentum_regressors(q[:, k],   dq[:, k])
        W1_k1, _  = quad_dyn.get_momentum_regressors(q[:, k+1], dq[:, k+1])
        W1_int += 0.5*(W1_k + W1_k1) * dt

    Y_h = (W2_ih - W2_i) - W1_int
    Y_h = P @ Y_h  # project

    # --- Compare predictions ---
    pred_classical = Y_int @ theta
    pred_momentum  = Y_h @ theta

    if joint_only:
        Sj = quad_dyn.S  # joints_dof x nv
        u_use = Sj @ u
        pc = Sj @ pred_classical
        pm = Sj @ pred_momentum
    else:
        u_use = u
        pc = pred_classical
        pm = pred_momentum

    e_classical = pc - u_use
    e_momentum  = pm - u_use
    e_diff      = pm - pc

    print("Window check:")
    print(f"  ||pred_classical - u|| = {np.linalg.norm(e_classical):.3e}")
    print(f"  ||pred_momentum  - u|| = {np.linalg.norm(e_momentum):.3e}")
    print(f"  ||pred_momentum - pred_classical|| = {np.linalg.norm(e_diff):.3e}")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # go back two levels to reach the root of the project
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    print(f"Project root directory: {dir_path}")
    urdf_path = os.path.join(dir_path, "files", "solo_description", "solo12.urdf")
    config_path = os.path.join(dir_path, "files", "solo_description", "solo12_config.yaml")

    q, dq, ddq, tau, cnt, force = load_data(dir_path, "trot_1khz")
    quad_dyn = QuadrupedDynamics(urdf_path, config_path)
    num_of_links = quad_dyn.get_num_links()
    phi_nominal = quad_dyn.get_phi_nominal()
    i = 2850
    dt = 0.001
    h_window = 90

    # check_classical_regressor(quad_dyn, q, dq, ddq, tau, cnt, force, i)
    # check_W2_regressor(quad_dyn, q, dq, i)
    # check_W1_regressor(quad_dyn, q, dq, i)
    # check_W1_with_dp(quad_dyn, q, dq, ddq, tau, cnt, force, dt, i)
    check_momentum_vs_classical_window(quad_dyn, q, dq, ddq, tau, cnt, force, dt, i, h_window, phi_nominal)