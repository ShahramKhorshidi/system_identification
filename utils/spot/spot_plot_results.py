import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.plot_calss import PlotClass
from src.sys_identification import SystemIdentification


def read_data(path, motion_name, data_noisy):
    start = 0
    end = 3000
    robot_q = np.loadtxt(path+f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_dq = np.loadtxt(path+f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_ddq = np.loadtxt(path+f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_tau = np.loadtxt(path+f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_contact = np.loadtxt(path+f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.int8)[:, start:end]
    tau_ped_nn = np.loadtxt(path+"tau_pred_nn.dat", delimiter='\t', dtype=np.float32).T[:, start:end]
    if data_noisy:
        # Butterworth filter parameters
        order = 5  # Filter order
        cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact, tau_ped_nn


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    
    motion_name = "spot_walk"
    q, dq, ddq, torque, cnt, tau_nn = read_data(path+"/data/spot/", motion_name, True)

    phi_prior = np.loadtxt(path+"/data/spot/spot_phi_prior.dat", delimiter='\t', dtype=np.float32)
    phi_proj_llsq = np.loadtxt(path+"/data/spot/spot_phi_proj_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_proj_lmi = np.loadtxt(path+"/data/spot/spot_phi_proj_lmi.dat", delimiter='\t', dtype=np.float32)
    b_v = np.loadtxt(path+"/data/spot/spot_b_v.dat", delimiter='\t', dtype=np.float32)
    b_c = np.loadtxt(path+"/data/spot/spot_b_c.dat", delimiter='\t', dtype=np.float32)
    print("b_v", b_v)
    print("b_c", b_c)
    # Instantiate the identification
    robot_urdf = path+"/files/spot_description/spot.urdf"
    robot_config = path+"/files/spot_description/spot_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    # Show Results
    sys_idnt.print_inertial_params(phi_prior, phi_proj_lmi)
    
    # Save new URDF file with identified parameters
    sys_idnt.update_urdf_inertial_params(phi_proj_llsq, b_v, b_c)
    
    # Plot physical consistency
    plotter = PlotClass(phi_prior)
    I_bar_prior, I_prior, J_prior, C_prior, trace_prior = sys_idnt.get_physical_consistency(phi_prior)
    plotter.plot_eigval(I_bar_prior, I_prior, J_prior, C_prior, trace_prior, "Phi Prior")
    
    I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq = sys_idnt.get_physical_consistency(phi_proj_llsq)
    plotter.plot_eigval(I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq, "Unconstrained llsq")

    I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi = sys_idnt.get_physical_consistency(phi_proj_lmi)
    plotter.plot_eigval(I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi, "Constrained LMI")
    
    # Plot inertial parameters
    plotter.plot_mass(phi_proj_llsq, "Projected llsq_Mass")
    plotter.plot_mass(phi_proj_lmi, "Projected LMI_Mass")
    
    plotter.plot_h(phi_proj_llsq, "Projected llsq_First Moment")
    plotter.plot_h(phi_proj_lmi, "Projected LMI_First moment")
    
    plotter.plot_inertia(phi_proj_llsq, "Projected llsq_Second Moment")
    plotter.plot_inertia(phi_proj_lmi, "Projected LMI_Second Moment")

    # Plot torques
    plotter.plot_proj_torques( q, dq, ddq, cnt, torque, b_v, b_c, phi_prior, sys_idnt, "Phi prior")
    tau_meas, tau_pred_llsq = plotter.plot_proj_torques(q, dq, ddq, cnt, torque, b_v, b_c, phi_proj_llsq, sys_idnt, "Projected LLSQ")
    tau_meas, tau_pred = plotter.plot_proj_torques(q, dq, ddq, cnt, torque, b_v, b_c, phi_proj_lmi, sys_idnt, "Projected LMI")
    plotter.plot_nn_torques(torque.T, tau_nn.T, "NN")
    # tau_pred = plotter.proj_torques_null(q, dq, ddq, cnt, tau_nn, b_v, b_c, phi_proj_lmi, sys_idnt)
    
    # Saving toruqes for plotting later for the paper, shape: (N_s,12)
    np.savetxt(path+"/data/spot/paper/"+f"{motion_name}_tau_meas.dat", tau_meas, delimiter='\t')
    np.savetxt(path+"/data/spot/paper/"+f"{motion_name}_tau_proj_lmi.dat", tau_pred, delimiter='\t')
    np.savetxt(path+"/data/spot/paper/"+f"{motion_name}_tau_proj_llsq.dat", tau_pred_llsq, delimiter='\t')
    np.savetxt(path+"/data/spot/paper/"+f"{motion_name}_tau_proj_nn.dat", tau_nn, delimiter='\t')
    
    plt.show()