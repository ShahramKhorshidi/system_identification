import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from src.solver import Solver
from src.sys_identification import SystemIdentification


def read_data(path, motion_name, filter_type):
    robot_q = np.loadtxt(path+motion_name+"_robot_q.dat", delimiter='\t', dtype=np.float32)
    robot_dq = np.loadtxt(path+motion_name+"_robot_dq.dat", delimiter='\t', dtype=np.float32)
    robot_ddq = np.loadtxt(path+motion_name+"_robot_ddq.dat", delimiter='\t', dtype=np.float32)
    robot_tau = np.loadtxt(path+motion_name+"_robot_tau.dat", delimiter='\t', dtype=np.float32)
    robot_contact = np.loadtxt(path+motion_name+"_robot_contact.dat", delimiter='\t', dtype=np.float32)
    
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
        window_length = 21  # window size (must be odd and greater than polyorder)
        polyorder = 10      # order of the polynomial fit
        # Apply Savitzky-Golay filter
        robot_dq = savgol_filter(robot_dq, window_length, polyorder)
        robot_ddq = savgol_filter(robot_ddq, window_length, polyorder)
        robot_tau = savgol_filter(robot_tau, window_length, polyorder)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def get_projected_y_tau(q, dq, ddq, torque, cnt, sys_idnt):
    Y = []
    Tau = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_proj_regressor_torque(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
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

def main():
    path = "/home/khorshidi/git/system_identification/"
    motion_name = "go1"
    filter_type = "butterworth" # "savitzky" "butterworth"
    q, dq, ddq, torque, cnt = read_data(path+"data/go1/", motion_name, filter_type)
    
    q = q[:, ::5]
    dq = dq[:, ::5]
    ddq = ddq[:, ::5]
    torque = torque[:, ::5]
    cnt = cnt[:, ::5] 
    
    robot_urdf = path+"files/go1_description/"+"go1.urdf"
    robot_config = path+"files/go1_description/"+"go1_config.yaml"
    
    # Instantiate the identification problem
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    total_mass = sys_idnt.get_robot_mass()
    num_of_links = sys_idnt.get_num_links()
    
    # Prior values for the inertial parameters
    phi_prior = sys_idnt.get_phi_prior()
    np.savetxt(path+"data/go1/"+"phi_prior.dat", phi_prior, delimiter='\t')
    
    # Bounding ellipsoids
    sys_idnt.compute_bounding_ellipsoids()
    bounding_ellipsoids = sys_idnt.get_bounding_ellipsoids()
    
    # -------- Using Null space projection -------- #
    Y_proj, Tau = get_projected_y_tau(q, dq, ddq, torque, cnt, sys_idnt)
    B_v, B_c = get_projected_friction_regressors(q, dq, ddq, cnt, sys_idnt)
    Y_proj = np.vstack(Y_proj)
    Tau = np.hstack(Tau)
    B_v = np.vstack(B_v)
    B_c = np.vstack(B_c)
    solver_proj = Solver(Y_proj, Tau, num_of_links, phi_prior, total_mass, bounding_ellipsoids) #, B_v=B_v, B_c=B_c)
    
    phi_proj_llsq = solver_proj.solve_llsq_svd()
    np.savetxt(path+"data/go1/"+motion_name+"_phi_proj_llsq.dat", phi_proj_llsq, delimiter='\t')
    
    phi_proj_lmi = solver_proj.solve_fully_consistent(lambda_reg=1e-2, epsillon=1e-4, max_iter=20000)
    np.savetxt(path+"data/go1/"+motion_name+"_phi_proj_lmi.dat", phi_proj_lmi, delimiter='\t')
    
    print("b_v:\n", b_v)
    print("b_c:\n", b_c)
if __name__ == "__main__":
    main()