import os
import pandas as pd
import numpy as np
import pinocchio as pin
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def preprocessing(n, path, motion_name):    
    # Read the CSV file into numpy arrays
    df = pd.read_csv(path+motion_name)
    
    t_s = df.iloc[:n, 0].to_numpy()
    t_ns = df.iloc[:n, 1].to_numpy()
    
    q_base_odom = df.iloc[:n, 2:9].to_numpy()
    q_base_vision = df.iloc[:n, 21:28].to_numpy()
    q_joints = df.iloc[:n, 9:21].to_numpy()
    
    dq_base_odom = df.iloc[:n, 40:46].to_numpy()
    dq_base_vision = df.iloc[:n, 58:64].to_numpy()
    dq_joints = df.iloc[:n, 46:58].to_numpy()
    
    ddq_base_odom = df.iloc[:n, 76:82].to_numpy()
    ddq_base_vision = df.iloc[:n, 94:100].to_numpy()
    ddq_joints = df.iloc[:n, 82:94].to_numpy()
    
    cnt = df.iloc[:n, 124:128].to_numpy()
    tau = df.iloc[:n, 112:124].to_numpy().T
    
    time = np.zeros((n), dtype=np.float32)
    q = np.zeros((19, n), dtype=np.float32)
    dq = np.zeros((18, n), dtype=np.float32)
    ddq = np.zeros((18, n), dtype=np.float32)
    contact = np.zeros((4, n), dtype=np.float32)
    
    for i in range(n):
        time[i] = t_s[i] - t_s[0] + t_ns[i] *1e-9
        
        # Robot configuration
        q[:7, i] = q_base_vision[i, :]
        q[7:, i] = q_joints[i, :]
        
        # Robot velocity
        dq[:6, i] = dq_base_vision[i, :]
        dq[6:, i] = dq_joints[i, :]
        
        # Robot acceleration
        ddq[:6, i] = ddq_base_vision[i, :]
        ddq[6:, i] = ddq_joints[i, :]
        for idx in range(4):
            if cnt[i, idx] == 1:
                contact[idx, i] = 1
    return time, q, dq, ddq, tau, contact

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
    cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)

    # Design Butterworth filter
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Apply Butterworth filter to each state (row in the data array)
    butter_signal = signal.filtfilt(b, a, orig_signal, axis=1)

    # Apply Savitzky-Golay filter
    window_length = 21  # window size (must be odd and greater than polyorder)
    polyorder = 10      # order of the polynomial fit

    savitzky_signal = savgol_filter(orig_signal, window_length, polyorder)

    # Plot the data
    fig, axs = plt.subplots(7, figsize=(10, 20))

    for i in range(7):
        j = i
        axs[i].plot(orig_signal[j, :],label='Original')
        axs[i].plot(butter_signal[j, :], label='Butter')
        # axs[i].plot(savitzky_signal[j, :], label='Savitzky-Golay')
        axs[i].set_xlabel('Sample')
        axs[0].legend()
    plt.tight_layout()
    plt.show()
    
def plot_2(data1, data2):
    # Plot two data arrays of the same dimension (Used for comparison of q_odom and q_vision)
    fig, axs = plt.subplots(6, figsize=(10, 20))

    for i in range(6):
        j = i
        axs[i].plot(data1[j, :],label='Odom')
        axs[i].plot(data2[j, :], label='Vision')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('')
        axs[0].legend()
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # Read the dats of different trajecories, "squat", "pose", "walk" and "crawl"
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parent_dir_path = os.path.dirname(dir_path) # Root directory of the workspace
    path = parent_dir_path+"/data/spot/"
    num_samples = 1000 # Number of samples for each trajectory
    time_0, q_0, dq_0, ddq_0, tau_0, cnt_0 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_squat.csv")
    time_1, q_1, dq_1, ddq_1, tau_1, cnt_1 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_pose_roll.csv")
    time_2, q_2, dq_2, ddq_2, tau_2, cnt_2 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_pose_pitch.csv")
    time_3, q_3, dq_3, ddq_3, tau_3, cnt_3 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_pose_yaw.csv")
    time_4, q_4, dq_4, ddq_4, tau_4, cnt_4 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_crawl_height_normal_speed_medium_sideways.csv")
    time_5, q_5, dq_5, ddq_5, tau_5, cnt_5 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_crawl_height_normal_speed_medium_forwardbackward.csv")
    time_6, q_6, dq_6, ddq_6, tau_6, cnt_6 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_walk_height_normal_speed_slow_forwardbackward.csv")
    time_7, q_7, dq_7, ddq_7, tau_7, cnt_7 = preprocessing(num_samples, path, motion_name="csv_files_2/spot_walk_height_normal_speed_fast_sideways.csv")
    
    # Concatenate data from all the trajectories into one array
    q = np.hstack((    q_0,   q_1,   q_2,   q_3,   q_4,   q_5,  q_6))
    dq = np.hstack((  dq_0,  dq_1,  dq_2,  dq_3,  dq_4,  dq_5,  dq_6))
    ddq = np.hstack((ddq_0, ddq_1, ddq_2, ddq_3, ddq_4, ddq_5, ddq_6))
    tau = np.hstack((tau_0, tau_1, tau_2, tau_3, tau_4, tau_5, tau_6))
    cnt = np.hstack((cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_5, cnt_6))
    
    # Shuffle the data
    X = np.concatenate((q, dq, ddq, tau, cnt), axis=0)
    print(X.shape)
    q = X[:19, :]
    dq = X[19:37, :]
    ddq = X[37:55, :]
    tau = X[55:67, :]
    contact = X[67:, :]
    
    # Save the combined trajectories into "spot" file
    # This data is used for inertial parameters identification
    np.savetxt(path+"spot_robot_q.dat", q, delimiter='\t')
    np.savetxt(path+"spot_robot_dq.dat", dq, delimiter='\t')
    np.savetxt(path+"spot_robot_ddq.dat", ddq, delimiter='\t')
    np.savetxt(path+"spot_robot_tau.dat", tau, delimiter='\t')
    np.savetxt(path+"spot_robot_contact.dat", cnt, delimiter='\t')
    
    # This data is used for validation with a new locomotion task 
    np.savetxt(path+"spot_walk_robot_q.dat", q_7, delimiter='\t')
    np.savetxt(path+"spot_walk_robot_dq.dat", dq_7, delimiter='\t')
    np.savetxt(path+"spot_walk_robot_ddq.dat", ddq_7, delimiter='\t')
    np.savetxt(path+"spot_walk_robot_tau.dat", tau_7, delimiter='\t')
    np.savetxt(path+"spot_walk_robot_contact.dat", cnt_7, delimiter='\t')
    
    # ddq_diff = finite_diff(time, dq)
    # plot(ddq)