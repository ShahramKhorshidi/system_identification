import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.animation import FuncAnimation
from src.sys_identification import SystemIdentification


# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'font.size' : 11,
#     'axes.unicode_minus' : False,
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

def read_data(path, motion_name, end, data_noisy):
    robot_q = np.loadtxt(path+f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)[:, :end]
    robot_dq = np.loadtxt(path+f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, :end]
    robot_ddq = np.loadtxt(path+f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, :end]
    robot_tau = np.loadtxt(path+f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, :end]
    robot_contact = np.loadtxt(path+f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.int8)[:, :end]
    if data_noisy:
        # Butterworth filter parameters
        order = 3  # Filter order
        cutoff_freq = 0.1  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
        # Apply Butterworth filter to each data (row in the data array)
        robot_dq = signal.filtfilt(b, a, robot_dq, axis=1)
        robot_ddq = signal.filtfilt(b, a, robot_ddq, axis=1)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

def compute_proj_torques(q, dq, ddq, torque, cnt, phi, sys_idnt):
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
    return measured, predicted 

def custom_y_formatter(x, pos):
    """Custom formatter to display y-axis labels without leading zeros for decimal numbers."""
    return f"{x:.2f}".lstrip('0').rstrip('0')


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    
    motion_name = "spot_validate"
    end = 1000
    q, dq, ddq, tau, cnt = read_data(path+"/data/spot/", motion_name, end, True)
    phi_prior = np.loadtxt(path+"/data/spot/spot_phi_prior.dat", delimiter='\t', dtype=np.float32)[:end]
    phi_proj_llsq = np.loadtxt(path+"/data/spot/spot_phi_proj_llsq.dat", delimiter='\t', dtype=np.float32)[:end]
    phi_proj_lmi = np.loadtxt(path+"/data/spot/spot_phi_proj_lmi.dat", delimiter='\t', dtype=np.float32)[:end]
    
    # Instantiate the identification
    robot_urdf = path+"/files/spot_description/spot.urdf"
    robot_config = path+"/files/spot_description/spot_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    tau_meas, tau_pred = compute_proj_torques(q, dq, ddq, tau, cnt, phi_proj_lmi, sys_idnt)
    tau_meas, tau_prior = compute_proj_torques(q, dq, ddq, tau, cnt, phi_prior, sys_idnt)
    # Downsampling
    rate = 1
    tau_meas = tau_meas[::rate, ]
    tau_pred = tau_pred[::rate, :]
    
    simulation_time = tau_meas.shape[0]
    print(simulation_time)

    fig, axs = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            # axs[i,j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            # axs[i,j].yaxis.set_major_formatter(FuncFormatter(custom_y_formatter))  # Apply custom formatting to y-axis
            axs[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))  # This line ensures x-axis ticks are integers

    t = np.arange(simulation_time) * rate / 100
    
    line_thick = 0.9
    
    # Indecis for motors in one leg
    i, j, k = 3, 4, 5
    
    # ----------------------- Hip Abduction/Adduction ----------------------- #
    # axs[0,0].plot(t, q[7+i, :], "k", linewidth=line_thick)
    axs[0,0].plot(t, tau_meas[:, i], "b", label="Meas", linewidth=line_thick)
    axs[0,0].plot(t, tau_prior[:, i], "r", label="Prior", linewidth=line_thick)
    # axs[0,0].plot(t, tau_pred[:, i], "y", label="LMI", linewidth=line_thick)
    axs[0,0].set(ylabel='Torque (Nm)')
    axs[0,0].get_yaxis().set_label_coords(-0.15,0.5)
    axs[0,0].set_xlabel('Hip Ab/Ad\n')
    axs[0,0].get_xaxis().set_label_coords(0.5,0.9)
    axs[0,0].xaxis.set_label_position('top')
    axs[0,0].xaxis.set_ticklabels([])
    axs[0,0].legend(loc="upper left", shadow=True, fontsize="xx-small")

    # ----------------------- Hip Flexion/Extension ----------------------- #
    axs[0,1].plot(t, tau_meas[:, j], "b", linewidth=line_thick)
    axs[0,1].plot(t, tau_prior[:, j], "r", linewidth=line_thick)
    # axs[0,1].plot(t, tau_pred[:, j], "y", linewidth=line_thick)
    axs[0,1].set_xlabel('Hip Fl/Ex\n')
    axs[0,1].get_xaxis().set_label_coords(0.5,0.9)  
    axs[0,1].xaxis.set_label_position('top')
    axs[0,1].xaxis.set_ticklabels([])
    axs[0,1].set_ylim([-15, 22])

    # ----------------------- Knee Flexion/Extension ----------------------- #
    axs[0,2].plot(t, tau_meas[:, k], "b", linewidth=line_thick)
    axs[0,2].plot(t, tau_prior[:, k], "r", linewidth=line_thick)
    # axs[0,2].plot(t, tau_pred[:, k], "y", linewidth=line_thick)
    axs[0,2].set_xlabel('Knee Fl/Ex\n')
    axs[0,2].get_xaxis().set_label_coords(0.5,0.9)    
    axs[0,2].xaxis.set_label_position('top')
    axs[0,2].xaxis.set_ticklabels([])

    # ----------------------- Hip Abduction/Adduction ----------------------- #
    axs[1,0].plot(t, tau_meas[:, i], "b", label="Meas", linewidth=line_thick)
    axs[1,0].plot(t, tau_pred[:, i], "y", label="LMI", linewidth=line_thick)
    axs[1,0].set(ylabel='Torque (Nm)')
    axs[1,0].get_yaxis().set_label_coords(-0.15,0.5)
    axs[1,0].set_xlabel('time (s)\n')
    # axs[1,0].get_xaxis().set_label_coords(0.5,0.9)
    axs[1,0].xaxis.set_label_position('bottom')
    # axs[1,0].xaxis.set_ticklabels([])
    axs[1,0].legend(loc="upper left", shadow=True, fontsize="xx-small")

    # ----------------------- Hip Flexion/Extension ----------------------- #
    axs[1,1].plot(t, tau_meas[:, j], "b", linewidth=line_thick)
    axs[1,1].plot(t, tau_pred[:, j], "y", linewidth=line_thick)
    axs[1,1].set_xlabel('time (s)\n')
    # axs[1,1].get_xaxis().set_label_coords(0.5,0.9)  
    axs[1,1].xaxis.set_label_position('bottom')
    # axs[1,1].xaxis.set_ticklabels([])
    axs[1,1].set_ylim([-15, 22])

    # ----------------------- Knee Flexion/Extension ----------------------- #
    axs[1,2].plot(t, tau_meas[:, k], "b", linewidth=line_thick)
    axs[1,2].plot(t, tau_pred[:, k], "y", linewidth=line_thick)
    axs[1,2].set_xlabel('time (s)\n')
    # axs[1,2].get_xaxis().set_label_coords(0.5,0.9)    
    axs[1,2].xaxis.set_label_position('bottom')
    # axs[1,2].xaxis.set_ticklabels([])

    # Show the plot
    fig.set_size_inches(w=8.2, h=3.2)
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.06, bottom=0.13, right=0.99, top=0.94, wspace=0.19, hspace=0.1)
    plt.show()

    # Save as pgf files
    # plt.savefig(path+"/data/spot/"+"spot_fl_torque.pgf")
