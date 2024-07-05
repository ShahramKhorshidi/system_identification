import numpy as np
from pathlib import Path
from src.solvers import Solvers
from src.sys_identification import SystemIdentification


def read_data(path):
    robot_q = np.loadtxt(path/"solo_robot_q.dat", delimiter='\t', dtype=np.float64)
    robot_dq = np.loadtxt(path/"solo_robot_dq.dat", delimiter='\t', dtype=np.float64)
    robot_ddq = np.loadtxt(path/"solo_robot_ddq.dat", delimiter='\t', dtype=np.float64)
    robot_tau = np.loadtxt(path/"solo_robot_tau.dat", delimiter='\t', dtype=np.float64)
    robot_contact = np.loadtxt(path/"solo_robot_contact.dat", delimiter='\t', dtype=np.float64)
    return robot_q, robot_dq, robot_ddq, robot_tau, robot_contact

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def calculate_regressor_and_torque(q, dq, ddq, torque, cnt, sys_idnt):
    Y = []
    Tau = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_regressor_pin(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
    return Y, Tau

def print_identified_parametrs(phi, num_of_links):
    total_mass = 0
    for i in range(num_of_links):
        print("### inertial Parameters of link", i, "###")
        index = 10*i
        print("mass:", phi[index])
        print("com:", phi[index+1: index+4]/phi[index])
        print("Inertia:", phi[index+4:index+10],"\n")
        total_mass += phi[index]
    print("### Total_mass:", total_mass, "###")

def main():
    path = Path.cwd()
    q, dq, ddq, torque, cnt = read_data(path/"data")
    robot_urdf = path/"files"/"solo12.urdf"
    robot_config = path/"files"/"solo12_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    # Calculate regressor and torque
    Y, Tau = calculate_regressor_and_torque(q, dq, ddq, torque, cnt, sys_idnt)
    Y = np.vstack(Y)
    Tau = np.hstack(Tau)
    
    # Instantiate the solver
    total_mass = sys_idnt.get_robot_mass()
    num_of_links =int(Y.shape[1]/10)
    # phi_prior is the prior values for the inertial parameters
    # It can be obtained from CAD file if available. Here I am jsut using some crude guess
    phi_prior = np.zeros(Y.shape[1])
    for i in range(num_of_links):
        j = 10 * i
        phi_prior[j] = 0.1
        phi_prior[j+1:j+4] = 3 * [0.1]
        phi_prior[j+4:j+10] = 6 * [0.005]
    
    bounding_ellipsoids = sys_idnt.get_bounding_ellipsoids()
    solver = Solvers(Y, Tau, phi_prior, bounding_ellipsoids)

    phi = solver.solve_fully_consistent(total_mass, lambda_reg=1e-1)
    print_identified_parametrs(phi, num_of_links)

if __name__ == "__main__":
    main()