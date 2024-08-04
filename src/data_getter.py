import numpy as np

class DataGetter:
    TIMESTAMP_LEN = 2
    POSITION_LEN = 19
    VELOCITY_LEN = 18
    ACCELERATION_LEN = 18
    LOAD_LEN = 12
    FOOT_STATE_LEN = 4

    def __init__(self, number_of_observations, percentage_for_testing):
        self.number_of_observations = number_of_observations
        self.observations_for_testing = int(number_of_observations * percentage_for_testing)

        self.j1 = 0       + self.TIMESTAMP_LEN
        self.j2 = self.j1 + self.POSITION_LEN
        self.j3 = self.j2 + self.POSITION_LEN
        self.j4 = self.j3 + self.VELOCITY_LEN
        self.j5 = self.j4 + self.VELOCITY_LEN
        self.j6 = self.j5 + self.ACCELERATION_LEN
        self.j7 = self.j6 + self.ACCELERATION_LEN
        self.j8 = self.j7 + self.LOAD_LEN
        self.j9 = self.j8 + self.FOOT_STATE_LEN

    def __process_data(self, timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, foot_state, data_haeder):
        # pinocchio needs contact_scedule instead of foot_state 
        vectorized_function = np.vectorize(lambda x: -x+2) # CONTACT_UNKNOWN:0->2, CONTACT_MADE:1->1, CONTACT_LOST:2->0
        contact_scedule = vectorized_function(foot_state.T)
        return timestamp, q_odom.T, q_vision.T, dq_odom.T, dq_vision.T, ddq_odom.T, ddq_vision.T, torque.T, contact_scedule, data_haeder

    def get_test_data_from_csv(self, data_path):
        data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
        data_haeder = np.genfromtxt(data_path, delimiter=',', skip_footer=len(data), dtype='str')

        timestamp  = data[:self.observations_for_testing,       0:self.j1]
        q_odom     = data[:self.observations_for_testing, self.j1:self.j2]
        q_vision   = data[:self.observations_for_testing, self.j2:self.j3]
        dq_odom    = data[:self.observations_for_testing, self.j3:self.j4]
        dq_vision  = data[:self.observations_for_testing, self.j4:self.j5]
        ddq_odom   = data[:self.observations_for_testing, self.j5:self.j6]
        ddq_vision = data[:self.observations_for_testing, self.j6:self.j7]
        torque     = data[:self.observations_for_testing, self.j7:self.j8]
        foot_state = data[:self.observations_for_testing, self.j8:self.j9]

        return self.__process_data(timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, foot_state, data_haeder)
    
    def get_validation_data_from_csv(self, data_path):
        data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
        data_haeder = np.genfromtxt(data_path, delimiter=',', skip_footer=len(data), dtype='str')

        timestamp  = data[self.observations_for_testing:,      0:self.j1]
        q_odom     = data[self.observations_for_testing:,self.j1:self.j2]
        q_vision   = data[self.observations_for_testing:,self.j2:self.j3]
        dq_odom    = data[self.observations_for_testing:,self.j3:self.j4]
        dq_vision  = data[self.observations_for_testing:,self.j4:self.j5]
        ddq_odom   = data[self.observations_for_testing:,self.j5:self.j6]
        ddq_vision = data[self.observations_for_testing:,self.j6:self.j7]
        torque     = data[self.observations_for_testing:,self.j7:self.j8]
        foot_state = data[self.observations_for_testing:,self.j8:self.j9]

        return self.__process_data(timestamp, q_odom, q_vision, dq_odom, dq_vision, ddq_odom, ddq_vision, torque, foot_state, data_haeder)