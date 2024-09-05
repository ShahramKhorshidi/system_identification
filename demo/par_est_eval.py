import os
import torch
import numpy as np
from src.PAR_EST import ESTIMATOR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Path to the trained model and the data
    dirPath = os.path.dirname(os.path.realpath(__file__))
    parentDirPath = os.path.dirname(dirPath)
    data_path = os.path.join(parentDirPath, "data/solo/")
    model_path = os.path.join(parentDirPath,"data/runs/Nets/340705.Sep.13.47_estimator_1000_1e-06", "epoch 15000.dat")

    # Load the data from the validation trajectory
    q = np.loadtxt(data_path + "nn_eval_robot_q.dat", delimiter='\t', dtype=np.float32)
    dq = np.loadtxt(data_path + "nn_eval_robot_dq.dat", delimiter='\t', dtype=np.float32)
    ddq = np.loadtxt(data_path + "nn_eval_robot_ddq.dat", delimiter='\t', dtype=np.float32)
    cnt = np.loadtxt(data_path + "nn_eval_robot_contact.dat", delimiter='\t', dtype=np.float32)
    tau = np.loadtxt(data_path + "nn_eval_robot_tau.dat", delimiter='\t', dtype=np.float32)
    
    # Stack the data into a single array
    X_val = np.vstack((q, dq, ddq, cnt))

    # Convert new data to PyTorch tensor and normalize it
    X_T = torch.tensor(X_val, dtype=torch.float32)
    U_T = torch.tensor(tau, dtype=torch.float32)
    X_T = X_T.permute(1,0)
    U_T = U_T.permute(1,0)
    
    # Replacing x and y position with delta_x and delta_y for each trajectory 
    X_T[1:2000, 0:2] = X_T[1:2000, 0:2] - X_T[0:1999, 0:2]
    X_T[0, 0:2] = 0

    # scale the input to be between -1 and 1 for each column
    X_T_MIN = X_T[:, :-4].min(0)[0]
    X_T_MAX = X_T[:, :-4].max(0)[0]
    U_T_MIN = U_T.min(0)[0]
    U_T_MAX = U_T.max(0)[0]
    X_T[:, :-4] = 2 * (X_T[:, :-4] - X_T_MIN) / (X_T_MAX - X_T_MIN) - 1
    print(torch.isnan(X_T).any())

    # Load the trained model
    u_train_shape = 12
    est = ESTIMATOR(X_T.shape[1], u_train_shape, False).to(device)
    est.load(model_path)

    # Move the new data to the appropriate device
    X_T = X_T.to(device)

    # Predict the torques
    tau_pred = est.cal_tau(X_T)
    
    # Convert the predicted torque back to numpy and unnormalize it
    tau_pred = ((tau_pred + 1) * (U_T_MAX.to(device) - U_T_MIN.to(device))) / 2 + U_T_MIN.to(device)  # Unnormalize the data
    tau_pred = tau_pred.cpu().numpy()
    
    # Save the predicted torque as a numpy file
    np.savetxt(data_path + "tau_pred_nn.dat", tau_pred, delimiter='\t')
