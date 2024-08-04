import numpy as np
import csv
from scipy.signal import butter, filtfilt
from pathlib import Path

def filter_base_acc_with_butter_worth_filter(qdd, timestamp):
    filtered_data = np.zeros((len(qdd), len(qdd[0])))
    
    filtered_data[:, 6:] = qdd[:, 6:]
    for i in range(0,6):
        # Design Butterworth filter
        filter_order = 4 # between 2 and 4 is used for smoothing data
        Wn = 0.3 # choose between 0.1 and 0.3 
        b, a = butter(N=filter_order, Wn=Wn, btype='low', analog=False)
        
        # data_frequency = 120.0 # Hz
        # nyquist_frequenc = 0.5 * data_frequency
        # critical_frequency = Wn * nyquist_frequenc # frequency at which the filter starts to attenuate the signal.

        # Apply the filter to the data
        filtered_data[:,i] = filtfilt(b, a, qdd[:,i])

    return filtered_data

def filter_vel_and_acc_and_load_with_butter_worth_filter(data):
    TIMESTAMP_LEN = 2
    POSITION_LEN = 19
    VELOCITY_LEN = 18
    ACCELERATION_LEN = 18
    LOAD_LEN = 12
    FOOT_STATE_LEN = 4
        
    j1 = 0  + TIMESTAMP_LEN
    j2 = j1 + POSITION_LEN
    j3 = j2 + POSITION_LEN
    j4 = j3 + VELOCITY_LEN
    j5 = j4 + VELOCITY_LEN
    j6 = j5 + ACCELERATION_LEN
    j7 = j6 + ACCELERATION_LEN
    j8 = j7 + LOAD_LEN
    j9 = j8 + FOOT_STATE_LEN

    timestamp  = data[:,  0:j1]
    q_odom     = data[:, j1:j2]
    q_vision   = data[:, j2:j3]
    dq_odom    = data[:, j3:j4]
    dq_vision  = data[:, j4:j5]
    ddq_odom   = data[:, j5:j6]
    ddq_vision = data[:, j6:j7]
    torque     = data[:, j7:j8]
    foot_state = data[:, j8:j9]
    
    filtered_data = np.zeros((len(data), len(data[0])))

    # Design Butterworth filter
    filter_order = 4 # between 2 and 4 is used for smoothing data
    Wn = 0.3 # choose between 0.1 and 0.3 
    b, a = butter(N=filter_order, Wn=Wn, btype='low', analog=False)
    # data_frequency = 120.0 # Hz
    # nyquist_frequenc = 0.5 * data_frequency
    # critical_frequency = Wn * nyquist_frequenc # frequency at which the filter starts to attenuate the signal.

    dq_odom_filtered = np.zeros((len(dq_odom),len(dq_odom[0])))
    dq_vision_filtered = np.zeros((len(dq_vision),len(dq_vision[0])))
    ddq_odom_filtered = np.zeros((len(ddq_odom),len(ddq_odom[0])))
    ddq_vision_filtered = np.zeros((len(ddq_vision),len(ddq_vision[0])))
    torque_filtered = np.zeros((len(torque),len(torque[0])))

    for i in range(0,VELOCITY_LEN):
        dq_odom_filtered[:,i]    = filtfilt(b, a, dq_odom[:,i])
        dq_vision_filtered[:,i]  = filtfilt(b, a, dq_vision[:,i])
    for i in range(0,ACCELERATION_LEN):
        ddq_odom_filtered[:,i]   = filtfilt(b, a, ddq_odom[:,i])
        ddq_vision_filtered[:,i] = filtfilt(b, a, ddq_vision[:,i])
    for i in range(0,LOAD_LEN):
        torque_filtered[:,i]     = filtfilt(b, a, torque[:,i])

    filtered_data[:,  0:j1] = timestamp
    filtered_data[:, j1:j2] = q_odom
    filtered_data[:, j2:j3] = q_vision
    filtered_data[:, j3:j4] = dq_odom_filtered
    filtered_data[:, j4:j5] = dq_vision_filtered
    filtered_data[:, j5:j6] = ddq_odom_filtered
    filtered_data[:, j6:j7] = ddq_vision_filtered
    filtered_data[:, j7:j8] = torque_filtered
    filtered_data[:, j8:j9] = foot_state

    return filtered_data

def print_filterd_data_to_file(data_haeder, data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data_haeder)
        for data_row in data:
            csv_writer.writerow(data_row)

if __name__ == "__main__":
    path = Path.cwd()
    path_crawl = str(path/"src"/"spot_crawl_speed_medium_height_low_graph_body_frame.csv")
    path_pose  = str(path/"src"/"spot_pose_graph_body_frame.csv")

    data = np.genfromtxt(path_pose, delimiter=',', skip_header=1)
    data_haeder = np.genfromtxt(path_pose, delimiter=',', skip_footer=len(data), dtype='str')
    data_filtered = filter_vel_and_acc_and_load_with_butter_worth_filter(data)
    print_filterd_data_to_file(data_haeder, data_filtered, "spot_pose_graph_body_frame_filtered.csv")