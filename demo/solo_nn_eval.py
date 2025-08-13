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
    model_path = os.path.join(parentDirPath,"data/runs/Nets/Samp_10000", "epoch 1000.dat")

    # Load the data from the validation trajectory
    motion_name = "eval_trot"
    start = 0
    end = 1000
    q = np.loadtxt(data_path + f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    dq = np.loadtxt(data_path + f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    ddq = np.loadtxt(data_path + f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    cnt = np.loadtxt(data_path + f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    tau = np.loadtxt(data_path + f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    
    # Stack the data into a single array
    X_val = np.vstack((q, dq, ddq, cnt))

    # Convert new data to PyTorch tensor and normalize it
    X_T = torch.tensor(X_val, dtype=torch.float32)
    U_T = torch.tensor(tau, dtype=torch.float32)
    X_T = X_T.permute(1,0)
    U_T = U_T.permute(1,0)
    
    # Replacing x and y position with delta_x and delta_y for each trajectory 
    X_T[1:end, 0:2] = X_T[1:end, 0:2] - X_T[0:end-1, 0:2]
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
    
    # Unnormalize the predicted torque and convert it back to numpy
    tau_pred = ((tau_pred + 1) * (U_T_MAX.to(device) - U_T_MIN.to(device))) / 2 + U_T_MIN.to(device)  # Unnormalize the data
    tau_pred = tau_pred.cpu().numpy()
    
    # Save the predicted torque as a numpy file
    error = tau.T - tau_pred
    rmse_total = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
    joint_tau_rmse = np.sqrt(np.mean(np.square(error), axis=0)) # RMSE for each joint
    print(f'\n-------------------- NN parameters --------------------')
    print(f'Torque Prediction Errors: RMSE_total= {rmse_total}\nRMSE_per_joints={joint_tau_rmse}')
    np.savetxt(data_path + "jump_tau_pred_nn.dat", tau_pred, delimiter='\t')
