import numpy as np
from pathlib import Path
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.plot_calss import PlotClass
from src.sys_identification import SystemIdentification


def read_data(path, motion_name, data_noisy):
    robot_q = np.loadtxt(path/f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)
    robot_dq = np.loadtxt(path/f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)
    robot_ddq = np.loadtxt(path/f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)
    robot_tau = np.loadtxt(path/f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)
    robot_ee_force = np.loadtxt(path/f"{motion_name}_robot_ee_force.dat", delimiter='\t', dtype=np.float32)
    robot_contact = np.loadtxt(path/f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.int8)
    if data_noisy:
        # Butterworth filter parameters
        order = 5  # Filter order
        cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_ee_force, robot_contact
    
    
if __name__ == "__main__":
    path = Path.cwd()
    
    motion_name = "noisy"
    q, dq, ddq, torque, force, cnt = read_data(path/"data"/"solo", motion_name, True)
    
    identified_params = "noisy"
    phi_prior = np.loadtxt(path/"data"/"solo"/"solo_phi_prior.dat", delimiter='\t', dtype=np.float32)
    phi_full_llsq = np.loadtxt(path/"data"/"solo"/f"{identified_params}_phi_full_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_full_lmi = np.loadtxt(path/"data"/"solo"/f"{identified_params}_phi_full_lmi.dat", delimiter='\t', dtype=np.float32)
    phi_proj_llsq = np.loadtxt(path/"data"/"solo"/f"{identified_params}_phi_proj_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_proj_lmi = np.loadtxt(path/"data"/"solo"/f"{identified_params}_phi_proj_lmi.dat", delimiter='\t', dtype=np.float32)
    
    # Instantiate the identification problem
    robot_urdf = path/"files"/"solo_description"/"solo12.urdf"
    robot_config = path/"files"/"solo_description"/"solo12_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    # Show Results
    sys_idnt.print_inertial_parametrs(phi_prior, phi_proj_lmi)
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, torque, cnt, phi_prior, "Prior")
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, torque, cnt, phi_full_llsq, "Full_llsq")
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, torque, cnt, phi_full_lmi, "Full_lmi")
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, torque, cnt, phi_proj_llsq, "Proj_llsq")
    sys_idnt.print_tau_prediction_rmse(q, dq, ddq, torque, cnt, phi_proj_lmi, "Proj_lmi")
    
    # Plot physical consistency
    plotter = PlotClass(phi_prior ,phi_proj_lmi)
    I_bar_prior, I_prior, J_prior, C_prior, trace_prior = sys_idnt.get_physical_consistency(phi_prior)
    plotter.plot_eigval(I_bar_prior, I_prior, J_prior, C_prior, trace_prior, "Phi Prior")
    
    I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq = sys_idnt.get_physical_consistency(phi_full_lmi)
    plotter.plot_eigval(I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq, "Full Sensing LMI")

    I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi = sys_idnt.get_physical_consistency(phi_proj_lmi)
    plotter.plot_eigval(I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi, "Projected LMI")
    
    # Plots
    plotter.plot_mass("Full Sensing LMI_Mass")
    plotter.plot_mass("Projected LMI_Mass")
    
    plotter.plot_h("Full Sensing LMI_First Moment")
    plotter.plot_h("Projected LMI_First moment")
    
    plotter.plot_inertia("Full Sensing LMI_Second Moment")
    plotter.plot_inertia("Projected LMI_Second Moment")

    plotter.plot_proj_torques(q, dq, ddq, torque, cnt, phi_full_lmi, sys_idnt, "Full Sensing")
    plotter.plot_proj_torques(q, dq, ddq, torque, cnt, phi_proj_lmi, sys_idnt, "Projected LMI")
    
    plt.show()