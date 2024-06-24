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

def main():
    path = Path.cwd()
    q, dq, ddq, torque, cnt = read_data(path/"data")
    sys_idnt = SystemIdentification(str(path/"urdf"/"solo12.urdf"), floating_base=True)
    
    # Calculate regressor and torque
    Y, Tau = calculate_regressor_and_torque(q, dq, ddq, torque, cnt, sys_idnt)
    Y = np.vstack(Y)
    Tau = np.hstack(Tau)
    
    # Solve the llsq problem
    solver = Solvers(Y, Tau)
    phi = solver.solve_llsq_svd()
    print("## Inertial parameters of the robot ##\n", phi)

if __name__ == "__main__":
    main()