import os
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from src.solver.lmi_solver import LMISolver
from src.dynamics.quadrupd_dynamics import QuadrupedDynamics


def read_data(path, motion_name, filter_type):
    robot_q = np.loadtxt(path+motion_name+"_robot_q.dat", delimiter='\t', dtype=np.float32)
    robot_dq = np.loadtxt(path+motion_name+"_robot_dq.dat", delimiter='\t', dtype=np.float32)
    robot_ddq = np.loadtxt(path+motion_name+"_robot_ddq.dat", delimiter='\t', dtype=np.float32)
    robot_tau = np.loadtxt(path+motion_name+"_robot_tau.dat", delimiter='\t', dtype=np.float32)
    robot_contact = np.loadtxt(path+motion_name+"_robot_contact.dat", delimiter='\t', dtype=np.float32)
    
    if filter_type=="butterworth":
        # Butterworth filter parameters
        order = 5  # Filter order
        cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
        robot_tau = signal.filtfilt(b, a, robot_tau, axis=1)
    elif filter_type=="savitzky":
        # Savitzky-Golay filter parameters
        window_length = 21  # window size (must be odd and greater than polyorder)
        polyorder = 10      # order of the polynomial fit
        # Apply Savitzky-Golay filter
        robot_dq = savgol_filter(robot_dq, window_length, polyorder)
        robot_ddq = savgol_filter(robot_ddq, window_length, polyorder)
        robot_tau = savgol_filter(robot_tau, window_length, polyorder)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def get_projected_y_tau(q, dq, ddq, torque, cnt, quad_dyn):
    Y = []
    Tau = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])
        y = quad_dyn.get_regressor_matrix(q[:, i], dq[:, i], ddq[:, i])
        P = quad_dyn.get_null_space_proj(cnt[:, i])
        Y.append(P @ y)
        Tau.append(P @ quad_dyn.S.T @ torque[:, i])
    return Y, Tau

# Calculates the friction regressors (B_v and B_c) projected into the null space of contact for all data points
def get_projected_friction_regressors(q, dq, ddq, cnt, quad_dyn):
    B_v = []
    B_c = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        quad_dyn.update_fk(q[:, i], dq[:, i], ddq[:, i])
        b_v, b_c = quad_dyn.get_friction_regressors(dq[:, i])
        P = quad_dyn.get_null_space_proj(cnt[:, i])
        B_v.append(P @ b_v)
        B_c.append(P @ b_c)
    return B_v, B_c

def main():
    # Read the data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    motion_name = "spot"
    filter_type = "butterworth" # savitzky or butterworth
    q, dq, ddq, torque, cnt = read_data(path+"/data/spot/", motion_name, filter_type)
    print(path)
    robot_urdf = path+"/files/spot_description/"+"spot.urdf"
    robot_config = path+"/files/spot_description/"+"spot_config.yaml"
    
    # Instantiate the quadruped dynamics and system identification objects
    quad_dyn = QuadrupedDynamics(robot_urdf, robot_config)
    
    total_mass = quad_dyn.get_robot_mass()
    num_of_links = quad_dyn.get_num_links()
    
    # Prior values for the inertial parameters
    phi_nominal = quad_dyn.get_phi_nom()
    np.savetxt(path+"/data/spot/"+"spot_phi_prior.dat", phi_nominal, delimiter='\t')
    
    # Bounding ellipsoids
    bounding_ellipsoids = quad_dyn.get_bounding_ellipsoids()
    
    # -------- Using Null space projection -------- #
    Y_proj, Tau = get_projected_y_tau(q, dq, ddq, torque, cnt, quad_dyn)
    B_v, B_c = get_projected_friction_regressors(q, dq, ddq, cnt, quad_dyn)
    Y_proj = np.vstack(Y_proj)
    Tau = np.hstack(Tau)
    B_v = np.vstack(B_v)
    B_c = np.vstack(B_c)
    lmi_solver = LMISolver(Y_proj, Tau, num_of_links, phi_nominal, total_mass, bounding_ellipsoids, B_v=B_v, B_c=B_c)
    
    phi_proj_llsq = lmi_solver.solve_llsq_svd()
    np.savetxt(path+"/data/spot/"+motion_name+"_phi_proj_llsq.dat", phi_proj_llsq, delimiter='\t')
    
    phi_proj_lmi, b_v, b_c = lmi_solver.solve_fully_consistent(lambda_reg=1e-3)
    np.savetxt(path+"/data/spot/"+motion_name+"_phi_proj_lmi.dat", phi_proj_lmi, delimiter='\t')
    np.savetxt(path+"/data/spot/"+motion_name+"_b_v.dat", b_v, delimiter='\t')
    np.savetxt(path+"/data/spot/"+motion_name+"_b_c.dat", b_c, delimiter='\t')

if __name__ == "__main__":
    main()