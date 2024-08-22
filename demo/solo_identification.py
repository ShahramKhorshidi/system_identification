import os
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from src.solver import Solver
from src.sys_identification import SystemIdentification


def read_data(path, filter_type):
    robot_q = np.loadtxt(path+"solo_robot_q.dat", delimiter='\t', dtype=np.float32)
    robot_dq = np.loadtxt(path+"solo_robot_dq.dat", delimiter='\t', dtype=np.float32)
    robot_ddq = np.loadtxt(path+"solo_robot_ddq.dat", delimiter='\t', dtype=np.float32)
    robot_tau = np.loadtxt(path+"solo_robot_tau.dat", delimiter='\t', dtype=np.float32)
    robot_contact = np.loadtxt(path+"solo_robot_contact.dat", delimiter='\t', dtype=np.float32)
    if filter_type=="butterworth":
        # Butterworth filter parameters
        order = 5  # Filter order
        cutoff_freq = 0.15  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
        robot_tau = signal.filtfilt(b, a, robot_tau, axis=1)
    elif filter_type=="savitzky":
        # Savitzky-Golay filter parameters
        polyorder = 5       # order of the polynomial fit
        window_length = 21  # window size (must be odd and greater than polyorder)
        # Apply Savitzky-Golay filter
        robot_dq = savgol_filter(robot_dq, window_length, polyorder)
        robot_ddq = savgol_filter(robot_ddq, window_length, polyorder)
        robot_tau = savgol_filter(robot_tau, window_length, polyorder)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def get_projected_y_tau(q, dq, ddq, torque, cnt, sys_idnt):
    Y = []
    Tau = []
    # For each data ponit we calculate the regressor and torque vector, and stack them
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_proj_regressor_torque(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
    return Y, Tau

# Calculates the friction regressors (B_v and B_c) projected into the null space of contact for all data points
def get_projected_friction_regressors(q, dq, ddq, cnt, sys_idnt):
    B_v = []
    B_c = []
    # For each data ponit we calculate friction regressors and stack them
    for i in range(q.shape[1]):
        b_v, b_c = sys_idnt.get_proj_friction_regressors(q[:, i], dq[:, i], ddq[:, i], cnt[:, i])
        B_v.append(b_v)
        B_c.append(b_c)
    return B_v, B_c

def main():
    path = os.getcwd()
    filter_type = "butterworth" # "savitzky" or "butterworth"
    q, dq, ddq, tau, cnt = read_data(path+"/data/solo/", filter_type)
    robot_urdf = path+"/files/solo_description/"+"solo12.urdf"
    robot_config = path+"/files/solo_description/"+"solo12_config.yaml"
    
    # Instantiate the identification problem
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    total_mass = sys_idnt.get_robot_mass()
    num_of_links = sys_idnt.get_num_links()
    
    # Prior values for the inertial parameters
    phi_prior = sys_idnt.get_phi_prior()
    
    # Bounding ellipsoids
    bounding_ellipsoids = sys_idnt.get_bounding_ellipsoids()
    
    # System Identification using null space projection
    # Calculate regressor and torque
    Y_proj, tau_proj = get_projected_y_tau(q, dq, ddq, tau, cnt, sys_idnt)
    B_v_proj, B_c_proj = get_projected_friction_regressors(q, dq, ddq, cnt, sys_idnt)
    Y_proj = np.vstack(Y_proj)
    tau_proj = np.hstack(tau_proj)
    B_v_proj = np.vstack(B_v_proj)
    B_c_proj = np.vstack(B_c_proj)
    
    # Solve the LMI and show the results
    solver_proj = Solver(Y_proj, tau_proj, num_of_links, phi_prior, total_mass, bounding_ellipsoids, B_v=B_v_proj, B_c=B_c_proj)
    phi_identified = solver_proj.solve_fully_consistent()
    sys_idnt.print_inertial_parametrs(phi_prior, phi_identified)
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_prior, "Prior")
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, tau, cnt, phi_identified, "Identified")


if __name__ == "__main__":
    main()