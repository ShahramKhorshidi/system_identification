import numpy as np
from pathlib import Path
from src.solver import Solver
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
        y, tau = sys_idnt.get_proj_regressor_torque(q[:, i], dq[:, i], ddq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
    return Y, Tau

def print_identified_parametrs(phi, num_of_links):
    total_mass = 0
    for i in range(num_of_links):
        print("### inertial Parameters of link", i, "###")
        index = 10*i
        print("Mass:", phi[index])
        print("CoM:", phi[index+1: index+4]/phi[index])
        I_xx, I_xy, I_xz, I_yy, I_yz, I_zz = phi[index+4:index+10]
        I_bar = np.array([[I_xx, I_xy, I_xz],
                          [I_xy, I_yy, I_yz],
                          [I_xz, I_yz, I_zz]])
        print("Inertia Matrix:\n", I_bar,"\n")
        total_mass += phi[index]
    print("### Total_mass:", total_mass, "###")

def main():
    path = Path.cwd()
    q, dq, ddq, torque, cnt = read_data(path/"data")
    robot_urdf = path/"files"/"solo12.urdf"
    robot_config = path/"files"/"solo12_config.yaml"
    
    # Instantiate the identification problem
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    total_mass = sys_idnt.get_robot_mass()
    num_of_links = sys_idnt.get_num_links()
    
    # Prior values for the inertial parameters
    phi_prior = sys_idnt.get_phi_prior()
    
    # Bounding ellipsoids
    bounding_ellipsoids = sys_idnt.get_bounding_ellipsoids()
    
    # Calculate regressor and torque
    Y, Tau = calculate_regressor_and_torque(q, dq, ddq, torque, cnt, sys_idnt)
    Y = np.vstack(Y)
    Tau = np.hstack(Tau)
    
    # Instantiate the solver
    solver = Solver(Y, Tau, num_of_links, phi_prior, total_mass, bounding_ellipsoids)

    phi = solver.solve_fully_consistent(lambda_reg=1e-2, epsillon=1e-3, max_iter=20000)
    print_identified_parametrs(phi, num_of_links)

if __name__ == "__main__":
    main()