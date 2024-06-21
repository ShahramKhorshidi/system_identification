import numpy as np
from pathlib import Path
from src.sys_identification import SystemIdentification
from src.solvers import Solvers


def read_data(path):
    robot_q = np.loadtxt(path/"solo_robot_q.dat", delimiter='\t', dtype=np.float64)
    robot_dq = np.loadtxt(path/"solo_robot_dq.dat", delimiter='\t', dtype=np.float64)
    robot_tau = np.loadtxt(path/"solo_robot_tau.dat", delimiter='\t', dtype=np.float64)
    robot_contact = np.loadtxt(path/"solo_robot_contact.dat", delimiter='\t', dtype=np.float64)
    return robot_q, robot_dq, robot_tau, robot_contact

# Calculates the regressor and torque vector projected into the null space of contact for all data points
def calculate_regressor_and_torque(q, dq, torque, cnt, sys_idnt):
    Y = []
    Tau = []
    for i in range(q.shape[1]):
        y, tau = sys_idnt.get_regressor_pin(q[:, i], dq[:, i], dq[:, i], torque[:, i], cnt[:, i])
        Y.append(y)
        Tau.append(tau)
    return Y, Tau


if __name__ == "__main__":
    path = Path.cwd()
    q, dq, torque, cnt = read_data(path/"data")

    sys_idnt = SystemIdentification(str(path/"urdf"/"solo12.urdf"), floating_base=True)
    # Calculate regressor and torque
    Y, Tau = calculate_regressor_and_torque(q, dq, torque, cnt, sys_idnt)    
    Y = np.vstack(Y)
    Tau = np.hstack(Tau)
    
    solver = Solvers(Y, Tau)
    phi = solver.solve_llsq_svd()
    print(phi)
