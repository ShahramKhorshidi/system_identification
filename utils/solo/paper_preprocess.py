import os
import torch
import numpy as np
from src.solver.nn_toque_estimator import ESTIMATOR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import scipy.signal as signal
from src.sys_identification import SystemIdentification

def calculate_rmse(q, dq, ddq, cnt, torque, b_v, b_c, phi, sys_idnt, title, force):
    predicted = []
    measured = []
    # For each data ponit we calculate the rgeressor and torque vector, and stack them
    for i in range(q.shape[1]):
        pred, meas = sys_idnt.calculate_predicted_torque_solo(q[:, i], dq[:, i], ddq[:, i], cnt[:, i], torque[:, i], b_v, b_c, phi, force[:, i])
        predicted.append(pred[6:])
        measured.append(meas[6:])
    
    predicted = np.vstack(predicted)
    measured = np.vstack(measured)
    
    error = measured - predicted
    rmse_total = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
    return rmse_total
        

def evaluate():
    start = 0
    end = 3000
    q = np.loadtxt(data_path + f"{motion_name}_robot_q.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    dq = np.loadtxt(data_path + f"{motion_name}_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    ddq = np.loadtxt(data_path + f"{motion_name}_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    cnt = np.loadtxt(data_path + f"{motion_name}_robot_contact.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    tau = np.loadtxt(data_path + f"{motion_name}_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    force = np.loadtxt(data_path+f"{motion_name}_robot_ee_force.dat", delimiter='\t', dtype=np.float32)[:, start:end]
    
    # MLP evaluation
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
    # print(torch.isnan(X_T).any())

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
    
    # Calculate the RMSE
    error = tau.T - tau_pred
    nn_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
    
    # LMI and SVD
    # Butterworth filter parameters
    order = 5  # Filter order
    cutoff_freq = 0.2  # Normalized cutoff frequency (0.1 corresponds to 0.1 * Nyquist frequency)
    # Design Butterworth filter
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    # Apply Butterworth filter to each data (row in the data array)
    dq = signal.filtfilt(b, a, dq, axis=1)
    ddq = signal.filtfilt(b, a, ddq, axis=1)
    
    phi_proj_llsq = np.loadtxt(lmi_path+"_phi_proj_llsq.dat", delimiter='\t', dtype=np.float32)
    phi_proj_lmi = np.loadtxt(lmi_path+"_phi_proj_lmi.dat", delimiter='\t', dtype=np.float32)
    b_v_proj = np.loadtxt(lmi_path+"_b_v_proj.dat", delimiter='\t', dtype=np.float32)
    b_c_proj = np.loadtxt(lmi_path+"_b_c_proj.dat", delimiter='\t', dtype=np.float32)
    
    llsq_rmse = calculate_rmse(q, dq, ddq, cnt, tau, b_v_proj, b_c_proj, phi_proj_llsq, sys_idnt, "Projected LLSQ", force)
    lmi_rmse = calculate_rmse(q, dq, ddq, cnt, tau, b_v_proj, b_c_proj, phi_proj_lmi, sys_idnt, "Projected LMI", force)
    return nn_rmse, llsq_rmse, lmi_rmse


if __name__ == "__main__":
    # Path to the trained model and the data
    dirPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parentDirPath = os.path.dirname(dirPath)
    data_path = os.path.join(parentDirPath, "data/solo/")
    
    # Instantiate the identification problem
    robot_urdf = parentDirPath+"/files/solo_description/solo12.urdf"
    robot_config = parentDirPath+"/files/solo_description/solo12_config.yaml"
    sys_idnt = SystemIdentification(str(robot_urdf), robot_config, floating_base=True)
    
    num_runs = 14
    num_eval = 10
    lmi_rmse = np.zeros((10), dtype=np.float32)
    svd_rmse = np.zeros((10), dtype=np.float32)
    nn_rmse = np.zeros((10), dtype=np.float32)
    
    nn = np.zeros((10), dtype=np.float32)
    llsq = np.zeros((10), dtype=np.float32)
    lmi = np.zeros((10), dtype=np.float32)
    
    # Load the data from the validation trajectories
    for j in range(1, 16): # number of models
        if j < 6 :
            eval = 100 * j
        else:
            eval = 1000 * (j-5)
        if j < 14:
            model_path = parentDirPath+"/data/runs/Nets/Samp_"+str(eval)+"/epoch 3000.dat"
        else:
            model_path = parentDirPath+"/data/runs/Nets/Samp_"+str(eval)+"/epoch 1000.dat"
        lmi_path = parentDirPath+"/data/runs/lmi/"+str(eval)
        print("\nj=", j)
        for i in range(9): # number of validation datasets used to finde average RMSE for each model
            print("i=", i)
            motion_name = "eval_"+str(i)
            rmse_nn, rmse_llsq, rmse_lmi = evaluate()
            nn[i] = rmse_nn
            llsq[i] = rmse_llsq
            lmi[i] = rmse_lmi
        # ---- This section should be only uncommented after the first run (100) was evaluated
        if i > 0:
            nn_rmse = np.vstack((nn_rmse, nn))
            svd_rmse = np.vstack((svd_rmse, llsq))
            lmi_rmse = np.vstack((lmi_rmse, lmi))

    np.savetxt(parentDirPath+"/data/solo/"+"Ave_RMSE_"+"nn.dat", nn_rmse, delimiter='\t')
    np.savetxt(parentDirPath+"/data/solo/"+"Ave_RMSE_"+"llsq.dat", svd_rmse, delimiter='\t')
    np.savetxt(parentDirPath+"/data/solo/"+"Ave_RMSE_"+"lmi.dat", lmi_rmse, delimiter='\t')