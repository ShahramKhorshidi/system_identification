from src.data_getter import DataGetter
from src.sys_identification import SystemIdentification
from src.filter_data import filter_base_acc_with_butter_worth_filter

import numpy as np
from pathlib import Path

class Validation:
    def __init__(self, number_of_observations, percentage_for_testing, trajectory_names, do_filter, do_odom):
        self.data_getter = DataGetter(number_of_observations, percentage_for_testing)      
        self.path = Path.cwd()
        self.sys_idnt = SystemIdentification(urdf_file = str(self.path/"files"/"spot.urdf"), config_file = str(self.path/"files"/"spot_config.yaml"), floating_base=True)
        self.Y_validation, self.Tau_validation = self.compute_inertial_parameters_for_some_trajectories_together(trajectory_names, do_filter, do_odom, "")

    # Calculates the regressor and torque vector projected into the null space of contact for all data points
    def compute_regressor_and_torque(self, data_path, file_name, Y, Tau, do_filter, do_odom):
        timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, contact_scedule, data_haeder = self.data_getter.get_validation_data_from_csv(data_path)

        if do_filter:
            ddq_odom = filter_base_acc_with_butter_worth_filter(ddq_odom, timestamp)
            ddq_vision = filter_base_acc_with_butter_worth_filter(ddq_vision, timestamp)

        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        if do_odom:
            for i in range(q_odom.shape[1]):
                y, tau = self.sys_idnt.get_proj_regressor_torque(q_odom[:, i], dq_odom[:, i], ddq_odom[:, i], torque[:, i], contact_scedule[:, i])
                Y.append(y)
                Tau.append(tau)
        else:
            for i in range(q_odom.shape[1]):
                y, tau = self.sys_idnt.get_proj_regressor_torque(q_vision[:, i], dq_vision[:, i], ddq_vision[:, i], torque[:, i], contact_scedule[:, i])
                Y.append(y)
                Tau.append(tau)
        
        print(f"Validation set: computed regressor and torque of data set {file_name}")
        return Y, Tau

    def compute_inertial_parameters_for_some_trajectories_together(self, file_names, do_filter, do_odom, trajectory_name):
        Y = []
        Tau = []
        for file_name in file_names:
            data_path = str(self.path/"data"/str(file_name + ".csv"))
            Y, Tau = self.compute_regressor_and_torque(data_path, file_name, Y, Tau, do_filter, do_odom)

        Y = np.vstack(Y) 
        Tau = np.hstack(Tau)
        
        return Y, Tau


    def direct_validation(self, phi, Y, Tau):
        return self.mean_squared_error(Y @ phi, Tau)

    def cross_validation(self, phi):
        return self.mean_squared_error(self.Y_validation @ phi, self.Tau_validation)

    def mean_squared_error(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError(f"len(x1)={len(x1)}!=len(x2)={len(x2)}")
        mse = 0
        for i in range(0, len(x1)):
            mse = mse + (x1[i]-x2[i])**2
        mse = mse / len(x1)
        return mse