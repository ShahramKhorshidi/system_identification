import numpy as np
from pathlib import Path
import scipy.signal as signal
import matplotlib.pyplot as plt
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
        order = 10  # Filter order
        cutoff_freq = 0.15  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
        robot_tau = signal.filtfilt(b, a, robot_tau, axis=1)
        robot_ee_force = signal.filtfilt(b, a, robot_ee_force, axis=1)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_ee_force, robot_contact

def plot_mass(phi_prior, calculated_phi, title):
    # Extract the mass values (every 10th element starting from the 0th element)
    masses_prior = phi_prior[::10]
    masses_calculated = calculated_phi[::10]

    # Create an array representing the link indices
    link_indices = np.arange(13)  # Links are numbered 1 to 13

    # Plot the bar diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.get_current_fig_manager().set_window_title(title)
    
    bar_width = 0.35  # Width of the bars
    index = np.arange(len(link_indices))

    bar1 = ax.bar(index, masses_prior, bar_width, label='Actual Mass')
    bar2 = ax.bar(index + bar_width, masses_calculated, bar_width, label='Identified Mass')

    # Adding labels, title and legend
    ax.set_xlabel('Link Number')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Masses for all links')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(link_indices)
    ax.legend()

    plt.tight_layout()

def plot_h(phi_prior, calculated_phi, title):
    num_links = 13
    num_params_per_link = 10
    
    # Initialize arrays to store h_x, h_y, h_z values
    h_x_actual = []
    h_y_actual = []
    h_z_actual = []
    h_x_predicted = []
    h_y_predicted = []
    h_z_predicted = []

    # Extract actual and predicted values
    for i in range(num_links):
        idx = i * num_params_per_link
        h_x_actual.append(phi_prior[idx + 1])
        h_y_actual.append(phi_prior[idx + 2])
        h_z_actual.append(phi_prior[idx + 3])
        
        h_x_predicted.append(calculated_phi[idx + 1])
        h_y_predicted.append(calculated_phi[idx + 2])
        h_z_predicted.append(calculated_phi[idx + 3])

    # Plotting
    index = np.arange(num_links)
    bar_width = 0.13

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    plt.get_current_fig_manager().set_window_title(title)
    
    # Plot h_x
    axs[0].bar(index - bar_width/2, h_x_actual, bar_width, label='Actual h_x')
    axs[0].bar(index + bar_width/2, h_x_predicted, bar_width, label='Identified h_x')
    axs[0].set_xticks(index)
    axs[0].set_xticklabels(range(num_links))
    axs[0].set_ylabel('h_x (kg.m)')
    axs[0].legend()

    # Plot x_y
    axs[1].bar(index - bar_width/2, h_y_actual, bar_width, label='Actual h_y')
    axs[1].bar(index + bar_width/2, h_y_predicted, bar_width, label='Identified h_h')
    axs[1].set_xticks(index)
    axs[1].set_xticklabels(range(num_links))
    axs[1].set_ylabel('h_y (kg.m)')
    axs[1].legend()

    # Plot h_z
    axs[2].bar(index - bar_width/2, h_z_actual, bar_width, label='Actual h_z')
    axs[2].bar(index + bar_width/2, h_z_predicted, bar_width, label='Identified h_z')
    axs[2].set_xticks(index)
    axs[2].set_xticklabels(range(num_links))
    axs[2].legend()
    axs[2].set_ylabel('h_z (kg.m)')
    axs[2].set_xlabel('Link Number')
    plt.tight_layout()

def plot_inertia(phi_prior, calculated_phi, title):
    num_links = 13
    num_params_per_link = 10
    
    # Initialize arrays to store h_x, h_y, h_z values
    I_xx_actual = []
    I_xy_actual = []
    I_xz_actual = []
    I_yy_actual = []
    I_yz_actual = []
    I_zz_actual = []
    I_xx_predicted = []
    I_xy_predicted = []
    I_xz_predicted = []
    I_yy_predicted = []
    I_yz_predicted = []
    I_zz_predicted = []

    # Extract actual and predicted values
    for i in range(num_links):
        idx = i * num_params_per_link
        I_xx_actual.append(phi_prior[idx + 4])
        I_xy_actual.append(phi_prior[idx + 5])
        I_xz_actual.append(phi_prior[idx + 6])
        I_yy_actual.append(phi_prior[idx + 7])
        I_yz_actual.append(phi_prior[idx + 8])
        I_zz_actual.append(phi_prior[idx + 9])
                
        I_xx_predicted.append(calculated_phi[idx + 4])
        I_xy_predicted.append(calculated_phi[idx + 5])
        I_xz_predicted.append(calculated_phi[idx + 6])
        I_yy_predicted.append(calculated_phi[idx + 7])
        I_yz_predicted.append(calculated_phi[idx + 8])
        I_zz_predicted.append(calculated_phi[idx + 9])
        
    # Plotting
    index = np.arange(num_links)
    bar_width = 0.13

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    plt.get_current_fig_manager().set_window_title(title)
    
    # Plot I_xx
    axs[0].bar(index - bar_width/2, I_xx_actual, bar_width, label='Actual I_xx')
    axs[0].bar(index + bar_width/2, I_xx_predicted, bar_width, label='Identified I_xx')
    # axs[0].set_xlabel('Link')
    # axs[0].set_ylabel('Values')
    # axs[0].set_title('I_xx for all links')
    axs[0].set_xticks(index)
    axs[0].set_xticklabels(range(num_links))
    axs[0].set_ylabel('I_xx (kg.m^2)')
    axs[0].legend()

    # Plot I_xy
    axs[1].bar(index - bar_width/2, I_xy_actual, bar_width, label='Actual I_xy')
    axs[1].bar(index + bar_width/2, I_xy_predicted, bar_width, label='Identified I_xy')
    axs[1].set_xticks(index)
    axs[1].set_xticklabels(range(num_links))
    axs[1].set_ylabel('I_xy (kg.m^2)')
    axs[1].legend()

    # Plot I_xz
    axs[2].bar(index - bar_width/2, I_xz_actual, bar_width, label='Actual I_xz')
    axs[2].bar(index + bar_width/2, I_xz_predicted, bar_width, label='Identified I_xz')
    axs[2].set_xticks(index)
    axs[2].set_xticklabels(range(num_links))
    axs[2].set_xlabel('Link Number')
    axs[2].set_ylabel('I_xz (kg.m^2)')
    axs[2].legend()
    
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 18))
    plt.get_current_fig_manager().set_window_title(title)
    
    # Plot I_yy
    axs2[0].bar(index - bar_width/2, I_yy_actual, bar_width, label='Actual I_yy')
    axs2[0].bar(index + bar_width/2, I_yy_predicted, bar_width, label='Identified I_yy')
    axs2[0].set_xticks(index)
    axs2[0].set_xticklabels(range(num_links))
    axs2[0].set_ylabel('I_yy (kg.m^2)')
    axs2[0].legend()

    # Plot I_yz
    axs2[1].bar(index - bar_width/2, I_yz_actual, bar_width, label='Actual I_yz')
    axs2[1].bar(index + bar_width/2, I_yz_predicted, bar_width, label='Identified I_yz')
    axs2[1].set_xticks(index)
    axs2[1].set_xticklabels(range(num_links))
    axs2[1].set_ylabel('I_yz (kg.m^2)')
    axs2[1].legend()

    # Plot I_zz
    axs2[2].bar(index - bar_width/2, I_zz_actual, bar_width, label='Actual I_zz')
    axs2[2].bar(index + bar_width/2, I_zz_predicted, bar_width, label='Identified I_zz')
    axs2[2].set_xticks(index)
    axs2[2].set_xticklabels(range(num_links))
    axs2[2].set_xlabel('Link Number')
    axs2[2].set_ylabel('I_zz (kg.m^2)')
    axs2[2].legend()

    plt.tight_layout()

def plot_proj_torques(q, dq, ddq, torque, cnt, phi, sys_idnt, title):
    predicted = []
    measured = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_proj_regressor_torque(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        pred = y@phi
        predicted.append(pred[6:])
        measured.append(tau[6:])
    
    predicted = np.vstack(predicted)
    measured = np.vstack(measured)
    
    num_joints = measured.shape[1]
    rows = 4
    cols = 3
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 15))
    plt.get_current_fig_manager().set_window_title(title)
    
    for j in range(num_joints):
        ax = axes[j // cols, j % cols]
        
        ax.plot(measured[:, j], label='Measured', color='blue')
        ax.plot(predicted[:, j], label='Identified', color='red', linestyle='--')
        
        if j == 0:
            ax.set_title("Hip Abduction/Adduction")
        if j == 1:
            ax.set_title("Hip Flexion/Extension")
        if j == 2:
            ax.set_title("Knee Flexion/Extension")
        ax.set_ylabel('Torque (Nm)')
        if j>=9:
            ax.set_xlabel('Sample')
        ax.legend()
    axes[0][0].set_ylabel('FL - Torque (Nm)')
    axes[1][0].set_ylabel('FR - Torque (Nm)')
    axes[2][0].set_ylabel('HL - Torque (Nm)')
    axes[3][0].set_ylabel('HR - Torque (Nm)')
    plt.tight_layout()
    
def plot_eigval(I_bar, I, J, C, trace, title):
    num_links = 13

    # Plot the bar diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.get_current_fig_manager().set_window_title(title)
    
    bar_width = 0.15  # Width of the bars
    index = np.arange(num_links)

    bar1 = ax.bar(index - bar_width, I_bar, bar_width, label='Inertia matrix')
    bar2 = ax.bar(index, I, bar_width, label='Spatial body inerta matrix')
    bar3 = ax.bar(index + bar_width, J, bar_width, label='Pseudo inertia matrix')
    bar4 = ax.bar(index + 2*bar_width, C, bar_width, label='CoM constraint matrix')
    # bar5 = ax.bar(index + 3*bar_width, trace, bar_width, label='trace')
    
    # Adding labels, title and legend
    ax.set_xlabel('Link Number')
    ax.set_ylabel('Minimum Eigen Value')
    ax.set_title('Minimum Eigenvalue for different matrices')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(index)
    ax.legend()
    plt.tight_layout()
    
    
if __name__ == "__main__":
    path = Path.cwd()
    
    motion_name = "solo"
    q, dq, ddq, torque, force, cnt = read_data(path/"data"/"solo", motion_name, False)
    
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
    I_bar_prior, I_prior, J_prior, C_prior, trace_prior = sys_idnt.get_physical_consistency(phi_prior)
    plot_eigval(I_bar_prior, I_prior, J_prior, C_prior, trace_prior, "Prior_physical consistency")
    
    I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq = sys_idnt.get_physical_consistency(phi_full_lmi)
    plot_eigval(I_bar_llsq, I_llsq, J_llsq, C_llsq, trace_llsq, "Full Sensing LMI")

    I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi = sys_idnt.get_physical_consistency(phi_proj_lmi)
    plot_eigval(I_bar_lmi, I_lmi, J_lmi, C_lmi, trace_lmi, "Projected LMI")
    
    # # Plots
    plot_mass(phi_prior, phi_full_lmi, "Full Sensing LMI_Mass")
    plot_mass(phi_prior, phi_proj_lmi, "Projected LMI_Mass")
    
    plot_h(phi_prior, phi_full_lmi, "Full Sensing LMI_First Moment")
    plot_h(phi_prior, phi_proj_lmi, "Projected LMI_First moment")
    
    plot_inertia(phi_prior, phi_full_lmi, "Full Sensing LMI_Second Moment")
    plot_inertia(phi_prior, phi_proj_lmi, "Projected LMI_Second Moment")

    plot_proj_torques(q, dq, ddq, torque, cnt, phi_full_lmi, sys_idnt, "Full Sensing")
    plot_proj_torques(q, dq, ddq, torque, cnt, phi_proj_lmi, sys_idnt, "Projected LMI")
    
    plt.show()