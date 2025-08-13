import torch
import numpy as np
import os, copy, errno
from src.PAR_EST import ESTIMATOR
import datetime
from src.PAR_EST import HID_DIM, LR

NETS = "estimator"
seed = 3407
# random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
date = datetime.datetime.now().strftime("%d.%h.%H.%M")
dirPath = os.path.dirname(os.path.realpath(__file__))
parentDirPath = os.path.dirname(dirPath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Read the data from file
    load_path = os.path.join(parentDirPath, "data/solo/")
    # q = np.loadtxt(load_path+"train2_robot_q.dat", delimiter='\t', dtype=np.float32)[:, :10000]
    # dq = np.loadtxt(load_path+"train2_robot_dq.dat", delimiter='\t', dtype=np.float32)[:, :10000]
    # ddq = np.loadtxt(load_path+"train2_robot_ddq.dat", delimiter='\t', dtype=np.float32)[:, :10000]
    # cnt = np.loadtxt(load_path+"train2_robot_contact.dat", delimiter='\t', dtype=np.float32)[:, :10000]
    # tau = np.loadtxt(load_path+"train2_robot_tau.dat", delimiter='\t', dtype=np.float32)[:, :10000]

    # X_t = q
    # X_t = np.vstack((X_t, dq))
    # X_t = np.vstack((X_t, ddq))
    # X_t = np.vstack((X_t, cnt))
    # U_t = tau
    
    # # Save the data as torch tensor
    # X_T = torch.tensor(X_t, dtype=torch.float32)
    # U_T = torch.tensor(U_t, dtype=torch.float32)
    # torch.save(X_T, load_path+"X_t_torch.pt")
    # torch.save(U_T, load_path+"U_t_torch.pt")
    
    # Load the tensor data
    X_T = torch.load(load_path+"X_t_torch.pt")
    U_T = torch.load(load_path+"U_t_torch.pt")
    
    # Test with a smaller data size of 10
    X_T = X_T.permute(1,0)#[:100, :]
    U_T = U_T.permute(1,0)#[:10, :]

    # Replacing x and y position with delta_x and delta_y for each trajectory
    # We have three trajectories each of the size N_s=3400
    N_s = 3400
    X_T[1:N_s, 0:2] = X_T[1:N_s, 0:2] - X_T[0:N_s-1, 0:2]
    X_T[0, 0:2] = 0
    
    X_T[N_s+1:2*N_s, 0:2] = X_T[N_s+1:2*N_s, 0:2] - X_T[N_s:2*N_s-1, 0:2]
    X_T[N_s, 0:2] = 0

    X_T[2*N_s+1:, 0:2] = X_T[2*N_s+1:, 0:2] - X_T[2*N_s:9999, 0:2]
    X_T[2*N_s, 0:2] = 0   

    # scale the data to be between -1 and 1 for each column
    X_T_MIN = X_T[:, :-4].min(0)[0]
    X_T_MAX = X_T[:, :-4].max(0)[0]
    U_T_MIN = U_T.min(0)[0]
    U_T_MAX = U_T.max(0)[0]
    
    X_T[:, :-4] = 2*(X_T[:, :-4] - X_T_MIN)/(X_T_MAX - X_T_MIN) - 1
    U_T = 2*(U_T - U_T_MIN)/(U_T_MAX - U_T_MIN) - 1
    
    print(torch.isnan(X_T).any())
    print(torch.isnan(U_T).any())
    
    # Shuffle the data
    idx = np.random.permutation(X_T.shape[0])
    X_t_norm = X_T[idx]
    U_t_norm = U_T[idx]
    
    # Change the sample size
    size = 10000 # Number of samples used for training
    X_t_norm = X_t_norm[:size, :]
    U_t_norm = U_t_norm[:size, :]
    batch_size = 256

    save_path = os.path.join(parentDirPath, "data/", "runs/Nets/"+str(seed)+str(date)+"_"+NETS+"_"\
        +str(HID_DIM)+"_NO_GAUS_"+str(LR)+"_"+str(batch_size))
    try:
        os.makedirs(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    # #reshape the data into -1, traj_length, and data_dim[1]
    # X_t_norm = X_T.view(-1, traj_length, X_T.shape[1])
    # U_t_norm = U_T.view(-1, traj_length, U_T.shape[1])
    
    #split the data into training and testing data 85% training and 15% testing
    train_size = int(X_t_norm.shape[0]*0.8)  
    
    X_train = X_t_norm[:train_size]
    U_train = U_t_norm[:train_size]
    X_test = X_t_norm[train_size:]
    U_test = U_t_norm[train_size:]
    
    est = ESTIMATOR(X_train.shape[1], U_train.shape[1], True).to(device)
    
    #create a loop to call the train function the model for 1000 epochs, each epoch will have 100 iterations. Save the model after every 1000 epochs
    iter = X_train.shape[0]//batch_size
    epochs = 5000
    for epoch in range(epochs):
        #print progress percentage the epoch number and delete the previous line
        print("Progress: %d%%, Epoch: %d, Nets: %s" % (epoch/epochs*100, epoch, NETS), end='\r')     
        for i in range(iter):
            # sample batch random indices from the training data and set them as the batch
            # idx = np.random.choice(X_train.shape[0], batch_size, replace=False)
            idx = np.arange(0,batch_size-1)
            X_train_batch = X_train[idx].to(device)
            U_train_batch = U_train[idx].to(device)
            est.train(X_train_batch, U_train_batch)
        
        if epoch > 4:
            #sample batch random indices from the testing data and set them as the batch
            idx = np.random.choice(X_test.shape[0], batch_size, replace=False)
            X_test_batch = X_test[idx].to(device)
            U_test_batch = U_test[idx].to(device)
            est.eval(X_test_batch, U_test_batch, epoch)
            if epoch % 1000 == 0 and epoch > 30:
                ename = "epoch %d.dat" % (epoch)
                fename = os.path.join(save_path, ename)
                est.save(fename)
