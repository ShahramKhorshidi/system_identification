import pandas as pd
import numpy as np
import pinocchio as pin
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def preprocessing(path, motion_name):    
    # Read the CSV file into numpy arrays
    df = pd.read_csv(path+motion_name)
    T = df.shape[0]
    
    q_base = df.iloc[:, 37:40].to_numpy()
    quat_x = df.iloc[:, 44].to_numpy()
    quat_y = df.iloc[:, 45].to_numpy()
    quat_z = df.iloc[:, 46].to_numpy()
    quat_w = df.iloc[:, 43].to_numpy()
    
    q_joints = df.iloc[:, 1:13].to_numpy()
    
    dq_base_lin = df.iloc[:, 40:43].to_numpy()
    dq_base_ang = df.iloc[:, 53:56].to_numpy()
    dq_joints = df.iloc[:, 13:25].to_numpy()
    
    ddq_base_lin = df.iloc[:, 50:53].to_numpy()
    
    # Finite differencing
    dq_base_dif = finite_diff(q_base[:, :].T)
    
    # cnt = df.iloc[:, 124:128].to_numpy()
    tau = df.iloc[:, 25:37].to_numpy().T
    
    q = np.zeros((19, T), dtype=np.float32)
    dq = np.zeros((18, T), dtype=np.float32)
    contact = np.ones((4, T), dtype=np.float32)
    
    for i in range(T):
        # Robot configuration
        # Base position
        q[:3, i] = q_base[i, :]
        # Base quaternion
        quat = pin.Quaternion(quat_w[i], quat_x[i], quat_y[i], quat_z[i])
        quat.normalize()
        q[3, i] = quat.x
        q[4, i] = quat.y
        q[5, i] = quat.z
        q[6, i] = quat.w
        # Joint positions
        q[7:, i] = q_joints[i, :]
        
        # Get the rotation matrix of the base
        R = quat.toRotationMatrix()

        # Robot velocity
        dq[0:3, i] = dq_base_lin[i, :] #R.T @ dq_base_dif[:, i] #dq_base_lin[i, :]
        dq[3:6, i] = dq_base_ang[i, :]
        dq[6:, i] = dq_joints[i, :]

    ddq = finite_diff(dq)
    for i in range(T):
        ddq[:3, i] = ddq_base_lin[i, :]
    
    # Robot acceleration
    q = q[:, ::5]
    dq = dq[:, ::5]
    ddq = ddq[:, ::5]
    tau = tau[:, ::5]
    contact = contact[:, ::5]    
    return q, dq, ddq, tau, contact

def finite_diff(dq):
    T = dq.shape[1]
    ddq = np.zeros((dq.shape[0], T), dtype=np.float32)
    
    # Butterworth filter parameters
    order = 5  # Filter order
    cutoff_freq = 0.15  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)

    # Design Butterworth filter
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Apply Butterworth filter to each state (row in the data array)
    dq_filt = signal.filtfilt(b, a, dq, axis=1)
    
    dt = 0.001 # sec
    dq_prev = dq_filt[:, 0]
    for i in range(1, T):
        dq_curr = dq_filt[:, i]
        ddq[:, i] = (dq_curr - dq_prev) / dt
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

    savitzky_signal = savgol_filter(orig_signal, window_length, polyorder)

    # Plot the data
    fig, axs = plt.subplots(6, figsize=(10, 20))

    for i in range(6):
        j = i
        axs[i].plot(orig_signal[j, :],label='Original')
        # axs[i].plot(butter_signal[j, :], label='Butter')
        # axs[i].plot(savitzky_signal[j, :], label='Savitzky-Golay')
        axs[i].set_xlabel('Sample')
        axs[0].legend()
    plt.tight_layout()
    plt.show()

def plot_2(data1, data2):
    # Plot two data arrays of the same dimension (Used for comparison)
    fig, axs = plt.subplots(3, figsize=(10, 20))

    for i in range(3):
        j = i
        axs[i].plot(data1[j, :],label='Fin_diff')
        axs[i].plot(data2[j, :], label='IMU')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('')
        axs[0].legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    path = "/home/khorshidi/git/system_identification/data/go1/"
    
    q_0, dq_0, ddq_0, tau_0, cnt_0 = preprocessing(path, motion_name="csv_files/wobbling_base.csv")
    # time_1, q_1, dq_1, ddq_1, tau_1, cnt_1 = preprocessing(path, motion_name="csv_files/go1_pose.csv")
    # time_2, q_2, dq_2, ddq_2, tau_2, cnt_2 = preprocessing(path, motion_name="csv_files/go1_walk_speed_slow_height_high_turn_around.csv")
    
    # q = np.hstack((q_0, q_1, q_2))
    # dq = np.hstack((dq_0, dq_1, dq_2))
    # ddq = np.hstack((ddq_0, ddq_1, ddq_2))
    # tau = np.hstack((tau_0, tau_1, tau_2))
    # cnt = np.hstack((cnt_0, cnt_1, cnt_2))
    
    np.savetxt(path+"go1_robot_q.dat", q_0, delimiter='\t')
    np.savetxt(path+"go1_robot_dq.dat", dq_0, delimiter='\t')
    np.savetxt(path+"go1_robot_ddq.dat", ddq_0, delimiter='\t')
    np.savetxt(path+"go1_robot_tau.dat", tau_0, delimiter='\t')
    np.savetxt(path+"go1_robot_contact.dat", cnt_0, delimiter='\t')
    
    plot(ddq_0)