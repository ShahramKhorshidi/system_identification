import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.plot_calss import PlotClass
from src.sys_identification import SystemIdentification


def read_data(path, motion_name, data_noisy):
    start = 0
    end = 1000
    robot_q = np.loadtxt(path+f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_dq = np.loadtxt(path+f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_ddq = np.loadtxt(path+f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_tau = np.loadtxt(path+f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    robot_ee_force = np.loadtxt(path+f"{motion_name}_robot_ee_force.dat", delimiter='\t', dtype=np.float32)[:, start:end]
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
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_ee_force, robot_contact, tau_ped_nn
    
    
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    
    # Load the trajectory and predicted tau_nn from the motion name
    motion_name = "eval_trot"
    q, dq, ddq, torque, force, cnt, tau_ped_nn = read_data(path+"/data/solo/", motion_name, True)

    # Load the identified model parameters optimized over noisy data
    identified_params = "train"
    phi_prior = np.loadtxt(path+"/data/solo/solo_phi_prior.dat", delimiter='\t', dtype=np.float32)
    phi_full_llsq = np.loadtxt(path+"/data/solo/"+f"{identified_params}_phi_full_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_full_lmi = np.loadtxt(path+"/data/solo/"+f"{identified_params}_phi_full_lmi.dat", delimiter='\t', dtype=np.float32)
    phi_proj_llsq = np.loadtxt(path+"/data/solo/"+f"{identified_params}_phi_proj_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_proj_lmi = np.loadtxt(path+"/data/solo/"+f"{identified_params}_phi_proj_lmi.dat", delimiter='\t', dtype=np.float32)
    b_v = np.loadtxt(path+"/data/solo/"+f"{identified_params}_b_v.dat", delimiter='\t', dtype=np.float32)
    b_c = np.loadtxt(path+"/data/solo/"+f"{identified_params}_b_c.dat", delimiter='\t', dtype=np.float32)
    b_v_proj = np.loadtxt(path+"/data/solo/"+f"{identified_params}_b_v_proj.dat", delimiter='\t', dtype=np.float32)
    b_c_proj = np.loadtxt(path+"/data/solo/"+f"{identified_params}_b_c_proj.dat", delimiter='\t', dtype=np.float32)

    # Instantiate the identification problem
    robot_urdf = path+"/files/solo_description/solo12.urdf"
    robot_config = path+"/files/solo_description/solo12_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    # Show Results
    sys_idnt.print_inertial_params(phi_prior, phi_proj_lmi)
    
    # Save new URDF file with identified parameters
    # sys_idnt.build_updated_urdf(phi_proj_lmi, b_v, b_c)
    
    # Plot physical consistency
    plotter = PlotClass(phi_prior)
    I_bar_prior, I_prior, J_prior, C_prior, trace_prior = sys_idnt.get_physical_consistency(phi_prior)
    plotter.plot_eigval(I_bar_prior, I_prior, J_prior, C_prior, trace_prior, "Phi Prior")
    
    I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq = sys_idnt.get_physical_consistency(phi_full_lmi)
    plotter.plot_eigval(I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq, "Full Sensing LMI")

    I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi = sys_idnt.get_physical_consistency(phi_proj_lmi)
    plotter.plot_eigval(I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi, "Projected LMI")
    
    # Plots
    plotter.plot_mass(phi_full_lmi, "Full Sensing LMI_Mass")
    plotter.plot_mass(phi_proj_lmi, "Projected LMI_Mass")
    
    plotter.plot_h(phi_full_lmi, "Full Sensing LMI_First Moment")
    plotter.plot_h(phi_proj_lmi, "Projected LMI_First moment")
    
    plotter.plot_inertia(phi_full_lmi, "Full Sensing LMI_Second Moment")
    plotter.plot_inertia(phi_proj_lmi, "Projected LMI_Second Moment")

    plotter.plot_solo_torques(q, dq, ddq, cnt, torque, b_v, b_c, phi_prior, sys_idnt, "Phi Prior", force)
    plotter.plot_solo_torques(q, dq, ddq, cnt, torque, b_v, b_c, phi_full_lmi, sys_idnt, "Full Sensing", force)
    tau_proj_llsq = plotter.plot_solo_torques(q, dq, ddq, cnt, torque, b_v_proj, b_c_proj, phi_proj_llsq, sys_idnt, "Projected LLSQ", force)
    tau_proj_lmi = plotter.plot_solo_torques(q, dq, ddq, cnt, torque, b_v_proj, b_c_proj, phi_proj_lmi, sys_idnt, "Projected LMI", force)
    
    # Normalize the measured torque if we want to compare against normalized output
    # tau_min = np.min(torque, axis=1)
    # tau_max = np.max(torque, axis=1)
    # torque = 2 * (torque - tau_min[:, None]) / (tau_max[:, None] - tau_min[:, None]) - 1
    plotter.plot_nn_torques(torque.T, tau_ped_nn.T, "NN")
    
    # Saving toruqes for plotting later for the paper, shape: (2000,12)
    # np.savetxt(path+"/data/solo/paper/"+f"{motion_name}_tau_meas.dat", torque, delimiter='\t')
    # np.savetxt(path+"/data/solo/paper/"+f"{motion_name}_tau_proj_llsq.dat", tau_proj_llsq, delimiter='\t')
    # np.savetxt(path+"/data/solo/paper/"+f"{motion_name}_tau_proj_lmi.dat", tau_proj_lmi, delimiter='\t')
    # np.savetxt(path+"/data/solo/paper/"+f"{motion_name}_tau_ped_nn.dat", tau_ped_nn.T, delimiter='\t')
    plt.show()