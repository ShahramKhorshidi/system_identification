import os
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from src.solver import Solver
from src.sys_identification import SystemIdentification


def read_data(path, motion_name, filter_type):
    start = 0
    end = 6000
    robot_q = np.loadtxt(path+motion_name+"_robot_q.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_dq = np.loadtxt(path+motion_name+"_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_ddq = np.loadtxt(path+motion_name+"_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_tau = np.loadtxt(path+motion_name+"_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_ee_force = np.loadtxt(path+motion_name+"_robot_ee_force.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_contact = np.loadtxt(path+motion_name+"_robot_contact.dat", delimiter='\t', dtype=np.float32)[:, start:end]
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
        robot_ee_force = signal.filtfilt(b, a, robot_ee_force, axis=1)
    elif filter_type=="savitzky":
        # Savitzky-Golay filter parameters
        window_length = 21  # window size (must be odd and greater than polyorder)
        polyorder = 5       # order of the polynomial fit
        
        # Apply Savitzky-Golay filter
        # robot_dq = savgol_filter(robot_dq, window_length, polyorder)
        robot_ddq = savgol_filter(robot_ddq, window_length, polyorder)
        robot_tau = savgol_filter(robot_tau, window_length, polyorder)
        robot_ee_force = savgol_filter(robot_ee_force, window_length, polyorder)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_ee_force, robot_contact

# Calculates the regressor and full force/torque vector
def get_full_y_f(q, dq, ddq, torque, force, cnt, sys_idnt):
    Y = []
    F = []
    # For each data ponit we calculate the rgeressor and force/torque vector, and stack them
    for i in range(q.shape[1]):
        y, f = sys_idnt.get_full_regressor_force(q[:, i], dq[:, i], ddq[:, i], torque[:, i], force[:, i], cnt[:, i])
        Y.append(y)
        F.append(f)
    return Y, F

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def get_projected_y_tau(q, dq, ddq, torque, cnt, sys_idnt):
    Y = []
    Tau = []
    tau_proj = np.zeros((12, q.shape[1]), dtype=np.float32)
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_proj_regressor_torque(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
        tau_proj[:, i] = tau[6:]
    return Y, Tau

# Calculates the friction regressors (B_v and B_c) projected into the null space of contact for all data points
def get_projected_friction_regressors(q, dq, ddq, cnt, sys_idnt):
    B_v = []
    B_c = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        b_v, b_c = sys_idnt.get_proj_friction_regressors(q[:, i], dq[:, i], ddq[:, i], cnt[:, i])
        B_v.append(b_v)
        B_c.append(b_c)
    return B_v, B_c

def get_full_friction_regressors(q, dq, ddq, cnt, sys_idnt):
    B_v = []
    B_c = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        b_v, b_c = sys_idnt.get_friction_regressors(q[:, i], dq[:, i], ddq[:, i], cnt[:, i])
        B_v.append(b_v)
        B_c.append(b_c)
    return B_v, B_c

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    motion_name = "train"
    filter_type = "butterworth" # "savitzky" "butterworth"
    q, dq, ddq, torque, force, cnt = read_data(path+"/data/solo/", motion_name, filter_type)
    robot_urdf = path+"/files/solo_description/"+"solo12.urdf"
    robot_config = path+"/files/solo_description/"+"solo12_config.yaml"
    
    # Instantiate the identification problem
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    total_mass = sys_idnt.get_robot_mass()
    num_of_links = sys_idnt.get_num_links()
    
    # Prior values for the inertial parameters
    phi_prior = sys_idnt.get_phi_prior()
    np.savetxt(path+"/data/solo/"+"solo_phi_prior.dat", phi_prior, delimiter='\t')
    phi_prior = np.loadtxt(path+"/data/solo/solo_phi_prior_I_c.dat", delimiter='\t', dtype=np.float32)
    
    # Bounding ellipsoids
    bounding_ellipsoids = sys_idnt.get_bounding_ellipsoids()
    
    # -------- Using full force/torque sensing -------- #
    Y, Force = get_full_y_f(q, dq, ddq, torque, force, cnt, sys_idnt)
    B_v, B_c = get_full_friction_regressors(q, dq, ddq, cnt, sys_idnt)
    Y = np.vstack(Y)
    Force = np.hstack(Force)
    B_v = np.vstack(B_v)
    B_c = np.vstack(B_c)
    solver_full = Solver(Y, Force, num_of_links, phi_prior, total_mass, bounding_ellipsoids, B_v=B_v, B_c=B_c)

    phi_full_llsq = solver_full.solve_llsq_svd()
    np.savetxt(path+"/data/solo/"+motion_name+"_phi_full_llsq.dat", phi_full_llsq, delimiter='\t')
    
    phi_full_lmi, b_v, b_c = solver_full.solve_fully_consistent()
    np.savetxt(path+"/data/solo/"+motion_name+"_phi_full_lmi.dat", phi_full_lmi, delimiter='\t')
    np.savetxt(path+"/data/solo/"+motion_name+"_b_v.dat", b_v, delimiter='\t')
    np.savetxt(path+"/data/solo/"+motion_name+"_b_c.dat", b_c, delimiter='\t')
    
    # -------- Using Null space projection -------- #
    Y_proj, tau_proj = get_projected_y_tau(q, dq, ddq, torque, cnt, sys_idnt)
    B_v_proj, B_c_proj = get_projected_friction_regressors(q, dq, ddq, cnt, sys_idnt)
    Y_proj = np.vstack(Y_proj)
    tau_proj = np.hstack(tau_proj)
    B_v_proj = np.vstack(B_v_proj)
    B_c_proj = np.vstack(B_c_proj)
    solver_proj = Solver(Y_proj, tau_proj, num_of_links, phi_prior, total_mass, bounding_ellipsoids, B_v=B_v_proj, B_c=B_c_proj)
    
    phi_proj_llsq = solver_proj.solve_llsq_svd()
    np.savetxt(path+"/data/solo/"+motion_name+"_phi_proj_llsq.dat", phi_proj_llsq, delimiter='\t')
    
    phi_proj_lmi, b_v_proj, b_c_proj = solver_proj.solve_fully_consistent()
    np.savetxt(path+"/data/solo/"+motion_name+"_phi_proj_lmi.dat", phi_proj_lmi, delimiter='\t')
    np.savetxt(path+"/data/solo/"+motion_name+"_b_v_proj.dat", b_v_proj, delimiter='\t')
    np.savetxt(path+"/data/solo/"+motion_name+"_b_c_proj.dat", b_c_proj, delimiter='\t')
    
if __name__ == "__main__":
    main()