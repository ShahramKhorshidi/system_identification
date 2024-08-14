import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def preprocessing():
    # Read the data.
    path = "/home/khorshidi/git/system_identification/data/spot/"
    
    # Read the CSV file into numpy arrays
    df = pd.read_csv(path+"spot_pose.csv")
    T = df.shape[0]
    
    t_s = df.iloc[:, 0].to_numpy()
    t_ns = df.iloc[:, 1].to_numpy()
    
    q_base_odom = df.iloc[:, 2:9].to_numpy()
    q_base_vision = df.iloc[:, 21:28].to_numpy()
    q_joints = df.iloc[:, 9:21].to_numpy()
    
    dq_base_odom = df.iloc[:, 40:46].to_numpy()
    dq_base_vision = df.iloc[:, 58:64].to_numpy()
    dq_joints = df.iloc[:, 46:58].to_numpy()
    
    ddq_base_odom = df.iloc[:, 76:82].to_numpy()
    ddq_base_vision = df.iloc[:, 94:100].to_numpy()
    ddq_joints = df.iloc[:, 82:94].to_numpy()
    
    time = np.zeros((T), dtype=np.float32)
    q = np.zeros((19, T), dtype=np.float32)
    dq = np.zeros((18, T), dtype=np.float32)
    ddq = np.zeros((18, T), dtype=np.float32)
    
    for i in range(T):
        time[i] = t_s[i] - t_s[0] + t_ns[i] *1e-9
        q[:7, i] = q_base_vision[i, :]
        q[7:, i] = q_joints[i, :]
        dq[:6, i] = dq_base_odom[i, :]
        dq[6:, i] = dq_joints[i, :]
        ddq[:6, i] = ddq_base_odom[i, :]
        ddq[6:, i] = ddq_joints[i, :]

    tau = df.iloc[:, 112:124].to_numpy().T
    cnt = df.iloc[:, 124:128].to_numpy().T
    
    return time, q, dq, ddq, tau, cnt

def finite_diff(time, dq):
    T = dq.shape[0]
    ddq = np.zeros((18, T), dtype=np.float32)
    
    dq_prev = dq[:, 0]
    for i in range(1, T):
        dt = time[i] - time[i-1]
        dq_curr = dq[:, i]
        ddq[i] = (dq_curr - dq_prev) / dt
        dq_prev = dq_curr
        
    return ddq
            
def plot(data):
    orig_signal = data
    # Butterworth filter parameters
    order = 5  # Filter order
    cutoff_freq = 0.15  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)

    # Design Butterworth filter
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Apply Butterworth filter to each state (row in the data array)
    butter_signal = signal.filtfilt(b, a, orig_signal, axis=1)

    # Apply Savitzky-Golay filter
    window_length = 21  # window size (must be odd and greater than polyorder)
    polyorder = 10      # order of the polynomial fit

    savavitzky_signal = savgol_filter(orig_signal, window_length, polyorder)

    # Plot the data
    fig, axs = plt.subplots(12, figsize=(10, 20))

    for i in range(12):
        j = i+6
        axs[i].plot(orig_signal[j, :],label='Original')
        axs[i].plot(butter_signal[j, :], label='Butter')
        # axs[i].plot(savavitzky_signal[j, :], label='Savitzky-Golay')
        axs[i].set_title(f'End Effector Force Dimension {i+1}')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Force')
        axs[0].legend()
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    time, q, dq, ddq, tau, cnt = preprocessing()
    path = "/home/khorshidi/git/system_identification/data/spot/"
    np.savetxt(path+"spot_robot_q.dat", q, delimiter='\t')
    np.savetxt(path+"spot_robot_dq.dat", dq, delimiter='\t')
    np.savetxt(path+"spot_robot_ddq.dat", ddq, delimiter='\t')
    np.savetxt(path+"spot_robot_tau.dat", tau, delimiter='\t')
    np.savetxt(path+"spot_robot_contact.dat", cnt, delimiter='\t')
    
    # ddq_diff = finite_diff(time, dq)
    plot(ddq)