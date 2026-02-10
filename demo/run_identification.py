import os
import argparse
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from src.solver.lmi_solver import LMISolver
from src.solver.nls_solver import NonlinearLeastSquares
from src.dynamics.quadrupd_dynamics import QuadrupedDynamics
from utils.plot_class import PlotClass
import matplotlib.pyplot as plt

def load_data(path, robot_name, filter_type):
    path = os.path.abspath(path) + os.sep

    robot_q = np.loadtxt(path + robot_name + "_robot_q.dat", delimiter="\t", dtype=np.float32)
    robot_dq = np.loadtxt(path + robot_name + "_robot_dq.dat", delimiter="\t", dtype=np.float32)
    robot_ddq = np.loadtxt(path + robot_name + "_robot_ddq.dat", delimiter="\t", dtype=np.float32)
    robot_tau = np.loadtxt(path + robot_name + "_robot_tau.dat", delimiter="\t", dtype=np.float32)
    robot_contact = np.loadtxt(path + robot_name + "_robot_contact.dat", delimiter="\t", dtype=np.float32)

    if filter_type == "butterworth":
        order = 5
        cutoff_freq = 0.15  # normalized (Nyquist=1.0)
        b, a = signal.butter(order, cutoff_freq, btype="low", analog=False)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
        robot_tau = signal.filtfilt(b, a, robot_tau, axis=1)

    elif filter_type == "savitzky":
        polyorder = 5
        window_length = 11  # must be odd and > polyorder
        robot_dq = savgol_filter(robot_dq, window_length, polyorder)
        robot_ddq = savgol_filter(robot_ddq, window_length, polyorder)
        robot_tau = savgol_filter(robot_tau, window_length, polyorder)

    elif filter_type == "none":
        pass
    else:
        raise ValueError(f"Unknown filter_type='{filter_type}'. Use: none | butterworth | savitzky")

    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

def get_projected_y_tau(q, dq, ddq, torque, cnt, quad_dyn):
    Y = []
    Tau = []
    for i in range(q.shape[1]):
        quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])
        y = quad_dyn.get_regressor_matrix(q[:, i], dq[:, i], ddq[:, i])
        # Here we project the regressor and torque into the null space of the contact constraints, 
        # If you use manipulator with no contact constraints, you should skip this projection.
        P = quad_dyn.get_null_space_proj(cnt[:, i])
        Y.append(P @ y)
        Tau.append(P @ quad_dyn.S.T @ torque[:, i])
    return Y, Tau

def get_projected_friction_regressors(q, dq, ddq, cnt, quad_dyn):
    B_v = []
    B_c = []
    for i in range(q.shape[1]):
        quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])
        b_v, b_c = quad_dyn.get_friction_regressors(dq[:, i])
        # Here we project the friction regressors into the null space of the contact constraints, 
        # If you use manipulator with no contact constraints, you should skip this projection.
        P = quad_dyn.get_null_space_proj(cnt[:, i])
        B_v.append(P @ b_v)
        B_c.append(P @ b_c)
    return B_v, B_c

def repo_root_from_this_file():
    # demo/run_identification.py -> repo root is one level up from demo/
    dir_path = os.path.dirname(os.path.realpath(__file__))   # .../demo
    return os.path.dirname(dir_path)                         # repo root

def get_robot_paths(root, robot):
    """
    Central place to map robot name -> urdf/config/description folder.
    Extend this dict as you add more robots.
    """
    robots = {
        "spot": {
            "description_dir": os.path.join(root, "files", "spot_description"),
            "urdf": os.path.join(root, "files", "spot_description", "spot.urdf"),
            "config": os.path.join(root, "files", "spot_description", "spot_config.yaml"),
            "mesh_dir": os.path.join(root, "files", "spot_description", "meshes", "base", "visual"),
        },
        # "go1": {...},
        # "anymal": {...},
    }
    if robot not in robots:
        raise ValueError(f"Unknown robot='{robot}'. Available: {list(robots.keys())}")
    return robots[robot]

def solve_lmi(q, dq, ddq, tau, cnt, quad_dyn):
    total_mass = quad_dyn.get_robot_mass()
    num_of_links = quad_dyn.get_num_links()
    phi_nominal = quad_dyn.get_phi_nominal()
    bounding_ellipsoids = quad_dyn.get_bounding_ellipsoids()

    Y_proj, tau_proj = get_projected_y_tau(q, dq, ddq, tau, cnt, quad_dyn)
    B_v_proj, B_c_proj = get_projected_friction_regressors(q, dq, ddq, cnt, quad_dyn)

    Y_proj = np.vstack(Y_proj)
    tau_proj = np.hstack(tau_proj)
    B_v_proj = np.vstack(B_v_proj)
    B_c_proj = np.vstack(B_c_proj)

    solver = LMISolver(
        Y_proj, tau_proj, num_of_links, phi_nominal, total_mass, bounding_ellipsoids,
        B_v=B_v_proj, B_c=B_c_proj
    )
    phi_identified, b_v, b_c = solver.solve_fully_consistent()

    # Reporting and plotting
    quad_dyn.print_inertial_params(phi_nominal, phi_identified)
    quad_dyn.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_nominal, "Nominal")
    quad_dyn.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_identified, "Identified", b_v, b_c, with_contact=True)

    plotter = PlotClass(phi_nominal)
    plotter.plot_mass(phi_identified, "Mass Comparison")
    plotter.plot_inertia(phi_identified, "Inertia Comparison")
    plt.show()

    return phi_identified

def solve_nls(q, dq, ddq, tau, cnt, quad_dyn):
    num_of_links = quad_dyn.get_num_links()
    phi_nominal = quad_dyn.get_phi_nominal()

    Y_proj, tau_proj = get_projected_y_tau(q, dq, ddq, tau, cnt, quad_dyn)
    B_v_proj, B_c_proj = get_projected_friction_regressors(q, dq, ddq, cnt, quad_dyn)

    Y_proj = np.vstack(Y_proj)
    tau_proj = np.hstack(tau_proj)
    B_v_proj = np.vstack(B_v_proj)
    B_c_proj = np.vstack(B_c_proj)

    solver = NonlinearLeastSquares(
        Y_proj, tau_proj, num_of_links, phi_nominal, B_v=B_v_proj, B_c=B_c_proj
    )
    phi_identified, b_v, b_c, _, _ = solver.solve_gn_exp(lambda_reg=1e-4, max_iters=100, tol=1e-5)

    # Reporting and plotting
    quad_dyn.print_inertial_params(phi_nominal, phi_identified)
    quad_dyn.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_nominal, "Nominal")
    quad_dyn.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_identified, "Identified", b_v, b_c, with_contact=True)

    plotter = PlotClass(phi_nominal)
    plotter.plot_mass(phi_identified, "Mass Comparison")
    plotter.plot_inertia(phi_identified, "Inertia Comparison")
    plt.show()

    return phi_identified

def parse_args():
    p = argparse.ArgumentParser(description="Offline inertial parameter identification demo")
    p.add_argument("--robot", type=str, default="spot", help="Robot name (e.g., spot)")
    p.add_argument("--solver", type=str, default="lmi", choices=["lmi", "nls"],
                   help="Solver backend: lmi (SDP) or nls (log-Cholesky)")
    p.add_argument("--filter", type=str, default="butterworth",
                   choices=["none", "butterworth", "savitzky"],
                   help="Pre-filter for dq/ddq/tau")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to data directory (defaults to <repo_root>/data)")
    return p.parse_args()

def main():
    args = parse_args()
    root = repo_root_from_this_file()

    data_dir = args.data_dir or os.path.join(root, "data")
    robot_paths = get_robot_paths(root, args.robot)

    # Load data
    q, dq, ddq, tau, cnt = load_data(data_dir, args.robot, args.filter)

    # Build dynamics model
    # You only need to provide the mesh_dir if you want to use the LMI solver.
    # For the NLS solver, you can set mesh_dir=None.
    mesh_dir = robot_paths["mesh_dir"] if args.solver == "lmi" else None
    quad_dyn = QuadrupedDynamics(robot_paths["urdf"], robot_paths["config"], mesh_dir)

    # Solve
    if args.solver == "lmi":
        solve_lmi(q, dq, ddq, tau, cnt, quad_dyn)
    elif args.solver == "nls":
        solve_nls(q, dq, ddq, tau, cnt, quad_dyn)
    else:
        raise ValueError(f"Unknown solver '{args.solver}'")


if __name__ == "__main__":
    main()
