
from bosdyn.api import robot_state_pb2

from bosdyn.client import create_standard_sdk
from bosdyn.client.sdk import Sdk
from bosdyn.client.robot import Robot
from bosdyn.client.robot_state import RobotStateClient

import csv
import numpy as np

NUMBER_OF_OBSERVATIONS = 5000

TIMESTAMP_LEN = 2
POSITION_LEN = 19
VELOCITY_LEN = 18
ACCELERATION_LEN = 18
LOAD_LEN = 12
FOOT_STATE_LEN = 4


def authenticate(ip, username = "admin", password = "password") -> Robot:
    print("ip: " + ip)
    print("username: " + username)
    print("password: " + password)
    
    sdk : Sdk = create_standard_sdk('ros_spot')
    robot : Robot = sdk.create_robot(ip)

    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()
    
    print("done authentication")
    return robot


def get_robot_state(robot_state_client : RobotStateClient, qd_odom_old, qd_vision_old, timestamp_old) -> robot_state_pb2.RobotState:
    # timestamp [s] and [10^{-9}s]
    # - seconds
    # - nanoseconds    

    # position [m]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z	body_ang_w
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # velocity [m/s]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # acceleration [m/s]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # loads [Nm]
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # foot_state []
    # - foot in contact: CONTACT_UNKNOWN=0, CONTACT_MADE=1, CONTACT_LOST=2

    # the base position and velocity can be measured in a odom or vison frame
    # the base acceleration can not be measured yet because the RobotStateStreamingService, which is needed to read the IMU data, is still in beta.
    # https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_state#bosdyn.client.robot_state.RobotStateStreamingClient.get_robot_state_stream
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#robotstatestreamingservice
    
    timestamp  = np.zeros((TIMESTAMP_LEN))
    q_odom     = np.zeros((POSITION_LEN))
    q_vision   = np.zeros((POSITION_LEN))
    qd_odom    = np.zeros((VELOCITY_LEN))
    qd_vision  = np.zeros((VELOCITY_LEN))
    qdd_odom   = np.zeros((ACCELERATION_LEN))
    qdd_vision = np.zeros((ACCELERATION_LEN))
    tau        = np.zeros((LOAD_LEN))
    foot_state = np.zeros((FOOT_STATE_LEN))

    robot_state = robot_state_client.get_robot_state()

    timestamp[0] = robot_state.kinematic_state.acquisition_timestamp.seconds
    timestamp[1] = robot_state.kinematic_state.acquisition_timestamp.nanos

    # base position
    q_odom[0] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.position.x
    q_odom[1] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.position.y
    q_odom[2] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.position.z
    q_odom[3] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.rotation.x
    q_odom[4] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.rotation.y
    q_odom[5] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.rotation.z
    q_odom[6] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("odom").parent_tform_child.rotation.w
    q_vision[0] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.position.x
    q_vision[1] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.position.y
    q_vision[2] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.position.z
    q_vision[3] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.rotation.x
    q_vision[4] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.rotation.y
    q_vision[5] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.rotation.z
    q_vision[6] = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("vision").parent_tform_child.rotation.w

    # base velocity
    qd_odom[0] = robot_state.kinematic_state.velocity_of_body_in_vision.linear.x
    qd_odom[1] = robot_state.kinematic_state.velocity_of_body_in_vision.linear.y
    qd_odom[2] = robot_state.kinematic_state.velocity_of_body_in_vision.linear.z
    qd_odom[3] = robot_state.kinematic_state.velocity_of_body_in_vision.angular.x
    qd_odom[4] = robot_state.kinematic_state.velocity_of_body_in_vision.angular.y
    qd_odom[5] = robot_state.kinematic_state.velocity_of_body_in_vision.angular.z
    qd_vision[0] = robot_state.kinematic_state.velocity_of_body_in_odom.linear.x
    qd_vision[1] = robot_state.kinematic_state.velocity_of_body_in_odom.linear.y
    qd_vision[2] = robot_state.kinematic_state.velocity_of_body_in_odom.linear.z
    qd_vision[3] = robot_state.kinematic_state.velocity_of_body_in_odom.angular.x
    qd_vision[4] = robot_state.kinematic_state.velocity_of_body_in_odom.angular.y
    qd_vision[5] = robot_state.kinematic_state.velocity_of_body_in_odom.angular.z
    
    # joint states: position, velocity, acceleration, load
    for js in range(0, len(robot_state.kinematic_state.joint_states)):
        q_odom[js+7]     = robot_state.kinematic_state.joint_states[js].position.value
        q_vision[js+7]   = robot_state.kinematic_state.joint_states[js].position.value
        qd_odom[js+6]    = robot_state.kinematic_state.joint_states[js].velocity.value
        qd_vision[js+6]  = robot_state.kinematic_state.joint_states[js].velocity.value
        qdd_odom[js+6]   = robot_state.kinematic_state.joint_states[js].acceleration.value
        qdd_vision[js+6] = robot_state.kinematic_state.joint_states[js].acceleration.value
        tau[js]          = robot_state.kinematic_state.joint_states[js].load.value

    # base and joint acceleration via finite differencing
    for i in range(0, ACCELERATION_LEN):
        delta_time_seconds = timestamp[0] - timestamp_old[0]
        delta_time_nanoseconds = timestamp[1] - timestamp_old[1]
        delta_time = delta_time_seconds + delta_time_nanoseconds*0.000000001
        delta_qd_odom = qd_odom[i] - qd_odom_old[i]
        delta_qd_vision = qd_vision[i] - qd_vision_old[i]
        if delta_time > 0:
            # base acceleration has to be filtered before data is used: butter worth filter?
            qdd_odom[i] = delta_qd_odom / delta_time
            qdd_vision[i] = delta_qd_vision / delta_time
        elif delta_qd_odom == 0 and delta_qd_vision == 0:
            qdd_odom[i] = 0.0
            qdd_vision[i] = 0.0
        elif delta_qd_odom == 0:
            qdd_odom[i] = 0.0
            qdd_vision[i] = float('NaN')
        elif delta_qd_vision == 0:
            qdd_odom[i] = float('NaN')
            qdd_vision[i] = 0.0       
        else:
            qdd_odom[i] = float('NaN')
            qdd_vision[i] = float('NaN')

    # foot_state
    for f in range(0, FOOT_STATE_LEN):
        foot_state[f] = robot_state.foot_state[f].contact

    return timestamp, q_odom, q_vision, qd_odom, qd_vision, qdd_odom, qdd_vision, tau, foot_state

def collect_data(robot_state_client : RobotStateClient, number_of_observations : int):
    timestamp_names       = np.array(["seconds", "nanoseconds"])
    joint_pos_names       = np.array(["joint_pos_fl_hx", "joint_pos_fl_hy", "joint_pos_fl_kn", "joint_pos_fr_hx", "joint_pos_fr_hy", "joint_pos_fr_kn", "joint_pos_hl_hx", "joint_pos_hl_hy", "joint_pos_hl_kn", "joint_pos_hr_hx", "joint_pos_hr_hy", "joint_pos_hr_kn"])
    joint_vel_names       = np.array(["joint_vel_fl_hx", "joint_vel_fl_hy", "joint_vel_fl_kn", "joint_vel_fr_hx", "joint_vel_fr_hy", "joint_vel_fr_kn", "joint_vel_hl_hx", "joint_vel_hl_hy", "joint_vel_hl_kn", "joint_vel_hr_hx", "joint_vel_hr_hy", "joint_vel_hr_kn"])
    joint_acc_names       = np.array(["joint_acc_fl_hx", "joint_acc_fl_hy", "joint_acc_fl_kn", "joint_acc_fr_hx", "joint_acc_fr_hy", "joint_acc_fr_kn", "joint_acc_hl_hx", "joint_acc_hl_hy", "joint_acc_hl_kn", "joint_acc_hr_hx", "joint_acc_hr_hy", "joint_acc_hr_kn"])
    joint_load_names      = np.array(["joint_load_fl_hx", "joint_load_fl_hy", "joint_load_fl_kn", "joint_load_fr_hx", "joint_load_fr_hy", "joint_load_fr_kn", "joint_load_hl_hx", "joint_load_hl_hy", "joint_load_hl_kn", "joint_load_hr_hx", "joint_load_hr_hy", "joint_load_hr_kn"])
    body_pos_names_odom   = np.array(["body_pos_lin_x_odom", "body_pos_lin_y_odom", "body_pos_lin_z_odom", "body_pos_ang_x_odom", "body_pos_ang_y_odom", "body_pos_ang_z_odom", "body_pos_ang_w_odom"])
    body_vel_names_odom   = np.array(["body_vel_lin_x_odom", "body_vel_lin_y_odom", "body_vel_lin_z_odom", "body_vel_ang_x_odom", "body_vel_ang_y_odom", "body_vel_ang_z_odom"])
    body_acc_names_odom   = np.array(["body_acc_lin_x_odom", "body_acc_lin_y_odom", "body_acc_lin_z_odom", "body_acc_ang_x_odom", "body_acc_ang_y_odom", "body_acc_ang_z_odom"])
    body_pos_names_vision = np.array(["body_pos_lin_x_vision", "body_pos_lin_y_vision", "body_pos_lin_z_vision", "body_pos_ang_x_vision", "body_pos_ang_y_vision", "body_pos_ang_z_vision", "body_pos_ang_w_vision"])
    body_vel_names_vision = np.array(["body_vel_lin_x_vision", "body_vel_lin_y_vision", "body_vel_lin_z_vision", "body_vel_ang_x_vision", "body_vel_ang_y_vision", "body_vel_ang_z_vision"])
    body_acc_names_vision = np.array(["body_acc_lin_x_vision", "body_acc_lin_y_vision", "body_acc_lin_z_vision", "body_acc_ang_x_vision", "body_acc_ang_y_vision", "body_acc_ang_z_vision"])
    foot_state_names      = np.array(["front_left_lower_leg", "front_right_lower_leg", "rear_left_lower_leg", "rear_right_lower_leg"])
    
    data_names = np.concatenate((timestamp_names, body_pos_names_odom, joint_pos_names, body_pos_names_vision, joint_pos_names, body_vel_names_odom, joint_vel_names, body_vel_names_vision, joint_vel_names, body_acc_names_odom, joint_acc_names, body_acc_names_vision, joint_acc_names, joint_load_names, foot_state_names))
    data = np.zeros((number_of_observations, len(data_names)))

    robot_state = robot_state_client.get_robot_state()
    start_seconds = robot_state.kinematic_state.acquisition_timestamp.seconds
    timestamp_old = np.zeros((TIMESTAMP_LEN))
    timestamp_old[0] = robot_state.kinematic_state.acquisition_timestamp.seconds
    timestamp_old[1] = robot_state.kinematic_state.acquisition_timestamp.nanos
    qd_odom_old   = np.zeros((VELOCITY_LEN))
    qd_vision_old = np.zeros((VELOCITY_LEN))

    j1 = 0  + TIMESTAMP_LEN
    j2 = j1 + POSITION_LEN
    j3 = j2 + POSITION_LEN
    j4 = j3 + VELOCITY_LEN
    j5 = j4 + VELOCITY_LEN
    j6 = j5 + ACCELERATION_LEN
    j7 = j6 + ACCELERATION_LEN
    j8 = j7 + LOAD_LEN
    j9 = j8 + FOOT_STATE_LEN

    print(f"start getting robot states...")

    for i in range(0, number_of_observations):
        timestamp, q_odom, q_vision, qd_odom, qd_vision, qdd_odom, qdd_vision, tau, foot_state = get_robot_state(robot_state_client, qd_odom_old, qd_vision_old, timestamp_old)

        data[i,  0:j1] = timestamp
        data[i, j1:j2] = q_odom
        data[i, j2:j3] = q_vision
        data[i, j3:j4] = qd_odom
        data[i, j4:j5] = qd_vision
        data[i, j5:j6] = qdd_odom
        data[i, j6:j7] = qdd_vision
        data[i, j7:j8] = tau
        data[i, j8:j9] = foot_state

        timestamp_old = timestamp
        qd_odom_old = qd_odom
        qd_vision_old = qd_vision

    print(f"got {number_of_observations} robot states in {timestamp[0] - start_seconds}sec")

    return data, data_names


def print_data_to_csv(data : np.matrix, data_names : np.matrix):
    filename = 'robot_state.csv'
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data_names)
        for row in data: # Writing data row by row
            csv_writer.writerow(row)
    print("Data has been written to", filename)


def main():
    robot : Robot = authenticate(ip = "192.168.80.3", username = "admin", password = "password")
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    data, data_names = collect_data(robot_state_client = robot_state_client, number_of_observations = NUMBER_OF_OBSERVATIONS)
    print_data_to_csv(data, data_names)


if __name__ == "__main__":
    main()