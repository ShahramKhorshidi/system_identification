from src.solvers import Solvers
from src.sys_identification import SystemIdentification

import numpy as np
import math
import csv
from pathlib import Path
import traceback

class SystemParameters:
    NUMBER_OF_OBSERVATIONS = 5000
    OBSERVATIONS_USED_FOR_ESTIMATION = 4000 # = NUMBER_OF_OBSERVATIONS * 0.8

    TIMESTAMP_LEN = 2
    POSITION_LEN = 19
    VELOCITY_LEN = 18
    ACCELERATION_LEN = 18
    LOAD_LEN = 12
    FOOT_STATE_LEN = 4

    NUMBER_OF_JOINTS = 12
    NUMBER_OF_LINKS = 13

    Y = []
    Tau = []

    def get_data_from_csv(self, filename, amount):
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        data_haeder = np.genfromtxt(filename, delimiter=',', skip_footer=len(data), dtype='str')

        j1 = 0  + self.TIMESTAMP_LEN
        j2 = j1 + self.POSITION_LEN
        j3 = j2 + self.POSITION_LEN
        j4 = j3 + self.VELOCITY_LEN
        j5 = j4 + self.VELOCITY_LEN
        j6 = j5 + self.ACCELERATION_LEN
        j7 = j6 + self.ACCELERATION_LEN
        j8 = j7 + self.LOAD_LEN
        j9 = j8 + self.FOOT_STATE_LEN

        timestamp  = data[:amount,0:j1]
        q_odom     = data[:amount,j1:j2]
        q_vision   = data[:amount,j2:j3]
        qd_odom    = data[:amount,j3:j4]
        qd_vision  = data[:amount,j4:j5]
        qdd_odom   = data[:amount,j5:j6]
        qdd_vision = data[:amount,j6:j7]
        torque     = data[:amount,j7:j8]
        foot_state = data[:amount,j8:j9]

        return timestamp, q_odom, q_vision, qd_odom, qd_vision, qdd_odom, qdd_vision, torque, foot_state, data_haeder

    # Calculates the regressor and torque vector projected into the null space of contact for all data points
    def compute_regressor_and_torque(self, data_path, sys_idnt, file_name):
        timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, foot_state, data_haeder = self.get_data_from_csv(data_path, amount=self.OBSERVATIONS_USED_FOR_ESTIMATION)

        # pinocchio needs contact_scedule instead of foot_state 
        vectorized_function = np.vectorize(lambda x: -x+2) # CONTACT_UNKNOWN:0->2, CONTACT_MADE:1->1, CONTACT_LOST:2->0
        contact_scedule = vectorized_function(foot_state.T)
        q = q_odom.T
        dq = dq_odom.T
        ddq = ddq_odom.T
        torque = torque.T

        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        for i in range(q.shape[1]):
            y, tau = sys_idnt.get_regressor_pin(q[:, 0], dq[:, 0], ddq[:, 0], torque[:, 0], contact_scedule[:, 0])
            self.Y.append(y)
            self.Tau.append(tau)
        
        print(f"computed regressor and torque of data set {file_name}")


    def compute_inertial_parameters(self):
        file_names = np.array(["all-random-1", "all-random-2", "all-random-3",
                            "crawl-circle-hight-high", "crawl-circle-hight-low", "crawl-circle-hight-normal",
                            "crawl-circle-speed-fast", "crawl-circle-speed-medium", "crawl-circle-speed-slow",
                            "crawl-random-hight", "crawl-random-speed",
                            "pose-random-1", "pose-random-2",
                            "pose-random-frontleft-elevated", "pose-random-frontright-elevated", "pose-random-rearleft-elevated", "pose-random-rearright-elevated",
                            "squat-1", "squat-2",
                            "stairs_speed_fast", "stairs_speed_medium", "stairs_speed_slow",
                            "walk-circle-hight-high", "walk-circle-hight-low", "walk-circle-hight-normal", 
                            "walk-circle-speed-fast", "walk-circle-speed-medium", "walk-circle-speed-slow",
                            "walk-random-hight", "walk-random-speed"])
        path = Path.cwd()
        sys_idnt = SystemIdentification(str(path/"urdf"/"spot_description"/"urdf"/"spot_org.urdf"), floating_base=True)

        for file_name in file_names:
            data_path = str(path/"data"/str(file_name + ".csv"))
            system_parameters.compute_regressor_and_torque(data_path, sys_idnt, file_name)

        print("start computing inertial parameters")

        Y = np.vstack(self.Y) 
        Tau = np.hstack(self.Tau)
        
        # Solve the llsq problem
        solver = Solvers(Y, Tau)
        phi = solver.solve_llsq_svd()

        # self.print_inertial_parameters(phi)
        data = self.print_inertial_parameters(phi)
        self.append_inertial_parameters_to_file(data, "all")

    def print_inertial_parameters(self, phi):
        inertial_names = np.array(["m", "mx", "my", "mz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]) # TODO order
        link_names = np.array(["body", 
                               "front_left_hip", "front_left_upper_leg", "front_left_lower_leg", 
                               "front_right_hip", "front_right_upper_leg", "front_right_lower_leg",
                               "rear_left_hip", "rear_left_upper_leg", "rear_left_lower_leg", 
                               "rear_right_hip", "rear_right_upper_leg", "rear_right_lower_leg"])
        
        data = np.empty((2, len(phi)+1), dtype=object)
        for i in range(0, len(phi)):
            data[0,i+1] = link_names[math.floor(i/13)] + "_" + inertial_names[i%10]
            data[1,i+1] = phi[i]
            print(f"{data[0,i+1]}: {data[1,i+1]}")
        return data

    def append_inertial_parameters_to_file(self, data, data_set_name):
        filename = 'inertial_parameters_spot.csv'
        data[1,0] = data_set_name
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for data_row in data:
                csv_writer.writerow(data_row)
        print(f"inertial parameters of data set {data_set_name} have been append to {filename}")
    
if __name__ == "__main__":
    system_parameters = SystemParameters()
    system_parameters.compute_inertial_parameters()
    
