import numpy as np
from pathlib import Path
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def preprocessing():
    # Read the data.
    path = "/home/khorshidi/git/system_identification/data/solo/"

    robot_q = []
    robot_dq = []
    robot_ddq = []
    robot_tau = []
    robot_ee_force = []
    robot_contact =[]
    
    for i in range(3):
        robot_q.append(np.loadtxt(path+str(i)+"_robot_q.dat", delimiter='\t', dtype=np.float64))
        robot_dq.append(np.loadtxt(path+str(i)+"_robot_dq.dat", delimiter='\t', dtype=np.float64))
        robot_ddq.append(np.loadtxt(path+str(i)+"_robot_ddq.dat", delimiter='\t', dtype=np.float64))
        robot_tau.append(np.loadtxt(path+str(i)+"_robot_tau.dat", delimiter='\t', dtype=np.float64))
        robot_ee_force.append(np.loadtxt(path+str(i)+"_robot_ee_force.dat", delimiter='\t', dtype=np.float64))
        robot_contact.append(np.loadtxt(path+str(i)+"_robot_contact.dat", delimiter='\t', dtype=np.float64))
    
    q = np.hstack(robot_q)
    dq = np.hstack(robot_dq)
    ddq = np.hstack(robot_ddq)
    tau = np.hstack(robot_tau)
    force = np.hstack(robot_ee_force)
    contact = np.hstack(robot_contact)
    
    np.savetxt(path+"solo_robot_q.dat", q, delimiter='\t')
    np.savetxt(path+"solo_robot_dq.dat", dq, delimiter='\t')
    np.savetxt(path+"solo_robot_ddq.dat", ddq, delimiter='\t')
    np.savetxt(path+"solo_robot_tau.dat", tau, delimiter='\t')
    np.savetxt(path+"solo_robot_ee_force.dat", force, delimiter='\t')
    np.savetxt(path+"solo_robot_contact.dat", contact, delimiter='\t')

def plot_data(motion_name):
    path = Path.cwd()/"data/solo/"
    robot_q = np.loadtxt(path/f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float64)
    robot_dq = np.loadtxt(path/f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float64)
    robot_ddq = np.loadtxt(path/f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float64)
    robot_tau = np.loadtxt(path/f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float64)
    robot_ee_force = np.loadtxt(path/f"{motion_name}_robot_ee_force.dat", delimiter='\t', dtype=np.float64)
    robot_contact = np.loadtxt(path/f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.int8)

    orig_signal = robot_tau
    # Butterworth filter parameters
    order = 5  # Filter order
    cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)

    # Design Butterworth filter
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Apply Butterworth filter to each state (row in the data array)
    butter_signal = signal.filtfilt(b, a, orig_signal, axis=1)

    # Apply Savitzky-Golay filter
    window_length = 35  # window size (must be odd and greater than polyorder)
    polyorder = 5      # order of the polynomial fit

    savitzky_signal = savgol_filter(orig_signal, window_length, polyorder)

    # Plot the data
    fig, axs = plt.subplots(12, figsize=(10, 20))

    for i in range(12):
        j = i
        axs[i].plot(orig_signal[j, :],label='Original' )
        axs[i].plot(butter_signal[j, :], label='Butter')
        # axs[i].plot(savitzky_signal[j, :], label='Savitzky-Golay')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Force')
        axs[0].legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # preprocessing()
    plot_data("noisy")