from src.solvers import Solvers
from src.data_getter import DataGetter
from src.sys_identification import SystemIdentification
from src.filter_data import filter_base_acc_with_butter_worth_filter
from src.validation import Validation

import numpy as np
import math
import csv
from pathlib import Path
import os

class SystemParameters:
    NUMBER_OF_OBSERVATIONS = 7000
    PERCENTAGE_FOR_TESTING = 0.8
    OBSERVATIONS_USED_FOR_ESTIMATION = PERCENTAGE_FOR_TESTING * NUMBER_OF_OBSERVATIONS # = 5600
    NUMBER_OF_LINKS = 13

    def __init__(self):
        self.data_getter = DataGetter(self.NUMBER_OF_OBSERVATIONS, self.PERCENTAGE_FOR_TESTING)
        self.all_file_names = np.array([
                                        "spot_crawl_speed_medium_height_high", "spot_crawl_speed_medium_height_low", "spot_crawl_speed_medium_height_normal",
                                        "spot_pose_frontleft_elevated", "spot_pose_frontright_elevated", "spot_pose_rearleft_elevated", "spot_pose_rearright_elevated", "spot_pose",
                                        "spot_random_1", "spot_random_2",
                                        # "spot_self_right", "spot_sit_stand", # do not use it because base is touching the floor
                                        "spot_squat",
                                        "spot_stairs_1", "spot_stairs_2", "spot_stairs_3",
                                        "spot_walk_speed_fast_height_high",   "spot_walk_speed_fast_height_low",   "spot_walk_speed_fast_height_normal",
                                        "spot_walk_speed_medium_height_high", "spot_walk_speed_medium_height_low", "spot_walk_speed_medium_height_normal",
                                        "spot_walk_speed_slow_height_high",   "spot_walk_speed_slow_height_low",   "spot_walk_speed_slow_height_normal"
                                        ])
        self.pose_file_names = np.array([
                                        "spot_pose_frontleft_elevated", "spot_pose_frontright_elevated", "spot_pose_rearleft_elevated", "spot_pose_rearright_elevated", "spot_pose",
                                        "spot_squat"
                                        ])
        self.pose_and_hight_high_file_names = np.array([
                                        "spot_crawl_speed_medium_height_high",
                                        "spot_pose_frontleft_elevated", "spot_pose_frontright_elevated", "spot_pose_rearleft_elevated", "spot_pose_rearright_elevated", "spot_pose",
                                        "spot_squat",
                                        "spot_walk_speed_fast_height_high", "spot_walk_speed_medium_height_high", "spot_walk_speed_slow_height_high",
                                        ])

        self.path = Path.cwd()
        self.sys_idnt = SystemIdentification(urdf_file=str(self.path/"urdf"/"spot_description"/"urdf"/"spot_org.urdf"), config_file = None, floating_base=True)

    # Calculates the regressor and torque vector projected into the null space of contact for all data points
    def compute_regressor_and_torque(self, data_path, file_name, Y, Tau, do_filter, do_odom):
        timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, contact_scedule, data_haeder = self.data_getter.get_test_data_from_csv(data_path)

        if do_filter:
            ddq_odom = filter_base_acc_with_butter_worth_filter(ddq_odom, timestamp)
            ddq_vision = filter_base_acc_with_butter_worth_filter(ddq_vision, timestamp)

        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        if do_odom:
            for i in range(q_odom.shape[1]):
                y, tau = self.sys_idnt.get_regressor_pin(q_odom[:, i], dq_odom[:, i], ddq_odom[:, i], torque[:, i], contact_scedule[:, i])
                Y.append(y)
                Tau.append(tau)
        else:
            for i in range(q_odom.shape[1]):
                y, tau = self.sys_idnt.get_regressor_pin(q_vision[:, i], dq_vision[:, i], ddq_vision[:, i], torque[:, i], contact_scedule[:, i])
                Y.append(y)
                Tau.append(tau)
        
        print(f"Test set:       computed regressor and torque of data set {file_name}")
        return Y, Tau

    def compute_inertial_parameters_for_some_trajectories_together(self, file_names, trajectory_name, do_filter, do_odom):
        Y = []
        Tau = []
        for file_name in file_names:
            data_path = str(self.path/"data"/str(file_name + ".csv"))
            Y, Tau = self.compute_regressor_and_torque(data_path, file_name, Y, Tau, do_filter, do_odom)

        Y = np.vstack(Y) 
        Tau = np.hstack(Tau)
        
        # Solve the llsq problem
        if do_filter:
            self.solve(Y, Tau, trajectory_name, "base acceleration filtered", file_names, do_filter, do_odom)
        else:
            self.solve(Y, Tau, trajectory_name, "nothing filtered", file_names, do_filter, do_odom)

    def compute_inertial_parameters_for_each_trajectory_independent(self, file_names, do_filter, do_odom):
        for file_name in file_names:
            data_path = str(self.path/"data"/str(file_name + ".csv"))
            Y = []
            Tau = []
            Y, Tau = self.compute_regressor_and_torque(data_path, file_name, Y, Tau, do_filter, do_odom)

            Y = np.vstack(Y) 
            Tau = np.hstack(Tau)
            
            # Solve the llsq problem
            if do_filter:
                self.solve(Y, Tau, file_name, "base acceleration filtered", np.array([file_name]), do_filter, do_odom)
            else:
                self.solve(Y, Tau, file_name, "nothing filtered", np.array([file_name]), do_filter, do_odom)

    def solve(self, Y, Tau, trajectory_name, filter_name, trajectory_names, do_filter, do_odom):
        bounding_ellipsoids = self.sys_idnt.get_bounding_ellipsoids()
        phi_prior = self.get_phi_prior(trajectory_name, filter_name, do_odom)
        solver = Solvers(Y, Tau, phi_prior, bounding_ellipsoids) # TODO for fully consistent
        phi_solver = [
                      solver.normal_equation,
                      solver.conjugate_gradient, 
                      solver.solve_llsq_svd, 
                    #   solver.wighted_llsq, 
                    #   solver.ridge_regression, 
                      solver.solve_semi_consistent, 
                      solver.solve_fully_consistent
                      ]
        
        validation = Validation(self.NUMBER_OF_OBSERVATIONS, self.PERCENTAGE_FOR_TESTING, trajectory_names, do_filter, do_odom)
        for solve_phi_function in phi_solver:
            solver_name = solve_phi_function.__name__
            print(f"start computing inertial parameters with {solver_name}")
            if solver_name == "solve_semi_consistent" or solver_name == "solve_fully_consistent":
                phi = solve_phi_function(total_mass=32.7)
            else:
                phi = solve_phi_function()
            direct_validation_rsme = validation.direct_validation(phi, Y, Tau)
            cross_validation_rsme = validation.cross_validation(phi)
            self.append_solution_to_file(trajectory_name, solver_name, filter_name, do_odom, direct_validation_rsme, cross_validation_rsme, phi)

    def append_solution_to_file(self, trajectory_name, solver_name, filter_name, do_odom, direct_validation_rsme, cross_validation_rsme, phi):
        filename = "results.csv"
        header = self.get_header()
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if os.stat(filename).st_size == 0:
                csv_writer.writerow(header)
            data = np.empty((self.NUMBER_OF_LINKS*10+6), dtype=object)
            data[0] = trajectory_name
            data[1] = solver_name
            data[2] = filter_name
            data[3] = "odom" if do_odom else "vision"
            data[4] = direct_validation_rsme
            data[5] = cross_validation_rsme
            data[6:] = phi
            csv_writer.writerow(data)
   
        print(f"inertial parameters of data set {trajectory_name} with solver {solver_name} have been append to {filename}")

    def get_header(self):
        inertial_names = np.array(["m", "mx", "my", "mz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]) # TODO order
        link_names = np.array(["body", 
                               "front_left_hip", "front_left_upper_leg", "front_left_lower_leg", 
                               "front_right_hip", "front_right_upper_leg", "front_right_lower_leg",
                               "rear_left_hip", "rear_left_upper_leg", "rear_left_lower_leg", 
                               "rear_right_hip", "rear_right_upper_leg", "rear_right_lower_leg"])
        
        header = np.empty((self.NUMBER_OF_LINKS*10+6), dtype=object)
        for i in range(0, self.NUMBER_OF_LINKS*10):
            header[i+6] = link_names[math.floor(i/10)] + "_" + inertial_names[i%10]
        header[0] = "trajectory_name"
        header[1] = "solver_name"
        header[2] = "filter_name"
        header[3] = "frame"
        header[4] = "direct_validation_rsme"
        header[5] = "cross_validation_rsme"
        return header 
    
    def get_phi_prior(self, trajectory_name, filter_name, do_odom):
        if do_odom: frame = "odom"
        else: frame = "vision"
        solver_name = "solve_semi_consistent"

        filename = "results.csv"
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == trajectory_name and row[1] == solver_name and row[2] == filter_name and row[3] == frame:
                    # row[4] == direct_validation_rsme, row[5] == cross_validation_rsme,
                    phi = np.array(row[6:130+6], dtype=np.float64)
                    return phi

        print(f"warning no phi was found for: {trajectory_name}, {solver_name}, {filter_name}, {frame} frame")
    
if __name__ == "__main__":
    system_parameters = SystemParameters()


    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_file_names, "pose_trajectories", do_filter=True, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_file_names, "pose_trajectories", do_filter=True, do_odom=False)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_file_names, "pose_trajectories", do_filter=False, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_file_names, "pose_trajectories", do_filter=False, do_odom=False)

    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_and_hight_high_file_names, "pose_and_hight_high_trajectories", do_filter=True, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_and_hight_high_file_names, "pose_and_hight_high_trajectories", do_filter=True, do_odom=False)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_and_hight_high_file_names, "pose_and_hight_high_trajectories", do_filter=False, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.pose_and_hight_high_file_names, "pose_and_hight_high_trajectories", do_filter=False, do_odom=False)

    system_parameters.compute_inertial_parameters_for_each_trajectory_independent(system_parameters.all_file_names, do_filter=True, do_odom=True)
    system_parameters.compute_inertial_parameters_for_each_trajectory_independent(system_parameters.all_file_names, do_filter=True, do_odom=False)
    system_parameters.compute_inertial_parameters_for_each_trajectory_independent(system_parameters.all_file_names, do_filter=False, do_odom=True)
    system_parameters.compute_inertial_parameters_for_each_trajectory_independent(system_parameters.all_file_names, do_filter=False, do_odom=False)

    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.all_file_names, "all_trajectories", do_filter=True, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.all_file_names, "all_trajectories", do_filter=True, do_odom=False)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.all_file_names, "all_trajectories", do_filter=False, do_odom=True)
    system_parameters.compute_inertial_parameters_for_some_trajectories_together(system_parameters.all_file_names, "all_trajectories", do_filter=False, do_odom=False)

    # system_parameters.compute_inertial_parameters_for_some_trajectories_together(np.array(["spot_squat"]), "spot_squat", do_filter=True, do_odom=True)
