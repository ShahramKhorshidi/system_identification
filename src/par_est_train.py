import torch
import numpy as np
import os, copy, errno
from PAR_EST import ESTIMATOR
import datetime

NETS = "estimator"
seed = 3407
# random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
date = datetime.datetime.now().strftime("%d.%h.%H.%M")
dirPath = os.path.dirname(os.path.realpath(__file__))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Read the data from files 1-5 and and merge them into one file
    # load_path = os.path.join(dirPath, "Files/")
    # for i in range(1,5):
    #     X = np.loadtxt(load_path+str(i)+"_cent_state"+".dat", delimiter='\t', dtype=np.float32).T
    #     U = np.loadtxt(load_path+str(i)+"_input_vec"+".dat", delimiter='\t', dtype=np.float32).T
    #     if i == 1:
    #         X_t = X
    #         U_t = U
    #     else:
    #         X_t = np.vstack((X_t, X))
    #         U_t = np.vstack((U_t, U))
    #18106000,
    # X = np.loadtxt(load_path+"trot_motion/trot_cent_state"+".dat", delimiter='\t', dtype=np.float32).T
    # U = np.loadtxt(load_path+"trot_motion/trot_input_vector"+".dat", delimiter='\t', dtype=np.float32).T
    # #17954000,
    # X = np.loadtxt(load_path+"jump_motion/jump_cent_state"+".dat", delimiter='\t', dtype=np.float32).T
    # U = np.loadtxt(load_path+"jump_motion/jump_input_vector"+".dat", delimiter='\t', dtype=np.float32).T
    # X_t = np.vstack((X_t, X))
    # U_t = np.vstack((U_t, U))
    # #9326000
    # X = np.loadtxt(load_path+"bound_motion/bound_cent_state"+".dat", delimiter='\t', dtype=np.float32).T
    # U = np.loadtxt(load_path+"bound_motion/bound_input_vector"+".dat", delimiter='\t', dtype=np.float32).T
    # X_t = np.vstack((X_t, X))
    # U_t = np.vstack((U_t, U))
    # #save the merged data   
    # print(X.shape)
    # print(U.shape)
    # X_T = torch.tensor(X, dtype=torch.float32)
    # U_T = torch.tensor(U, dtype=torch.float32)
    # torch.save(X_T, "X_t_t.pt")
    # torch.save(U_T, "U_t_t.pt")
    # quit()
    # #load the merged data
    # X_t = np.loadtxt("X_t_merged.dat", delimiter='\t', dtype=np.float64)
    # U_t = np.loadtxt("U_t_merged.dat", delimiter='\t', dtype=np.float64)
    # X_t_ten = torch.tensor(X_t, dtype=torch.float32)
    # U_t_ten = torch.tensor(U_t, dtype=torch.float32)
    STEPS_FOR_A = 10
    X_T = torch.load("X_t_20.pt")
    U_T = torch.load("U_t_20.pt") #10782000 size

    # size = X_T_j.shape[0]
    # X_T = X_T_j[:size//3]
    # U_T = U_T_j[:size//3]
    # X_T_t = torch.load("X_trot.pt")
    # U_T_t = torch.load("U_trot.pt") #10782000 size
    # X_T = torch.cat((X_T, X_T_t[:size//3]))
    # U_T = torch.cat((U_T, U_T_t[:size//3]))
    traj_length = 200
    X_T = X_T.view(-1, traj_length, X_T.shape[1])
    init_pos = X_T[:,0,:2]
    X_T[:, :, :2] -= init_pos.repeat(1, traj_length).view(-1,traj_length, 2)
    X_T = X_T.view(-1,9)

    
    # Save the min and max values of the data excpet the first and second columns of X_T
    X_T_MIN = X_T.min(0)[0]
    X_T_MAX = X_T.max(0)[0]
    U_T_MIN = U_T.min(0)[0]
    U_T_MAX = U_T.max(0)[0]
    X_T= 2*(X_T - X_T_MIN)/(X_T_MAX - X_T_MIN) - 1
    U_T = 2*(U_T - U_T_MIN)/(U_T_MAX - U_T_MIN) - 1
    
    #scale the data to be between -1 and 1 for each column
    print(torch.isnan(X_T).any())
    print(torch.isnan(U_T).any())

    batch_size = 64
    # pretrain_steps = 5000
    
    save_path = os.path.join(dirPath, "runs/Nets/"+str(seed)+str(date)+"_"+NETS+"_")
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    
    #reshape the data into -1, traj_length, and data_dim[1]
    X_t_norm = X_T.view(-1, traj_length, X_T.shape[1])
    U_t_norm = U_T.view(-1, traj_length, U_T.shape[1])
    
    #shuffle the data
    idx = np.random.permutation(X_t_norm.shape[0])
    X_t_norm = X_t_norm[idx]
    U_t_norm = U_t_norm[idx]
    
    #split the data into training and testing data 85% training and 15% testing
    train_size = int(X_t_norm.shape[0]*0.8)  
    
    X_train = X_t_norm[:train_size]
    U_train = U_t_norm[:train_size]
    X_test = X_t_norm[train_size:]
    U_test = U_t_norm[train_size:]
    
    
    est = ESTIMATOR(X_train.shape[2], U_train.shape[2]).to(device)
    
    
    #create a loop to call the train function the model for 1000 epochs, each epoch will have 100 iterations. Save the model after every 1000 epochs
    
    
    iter = X_train.shape[0]//batch_size
    epochs = 200000
    for epoch in range(epochs):
        #print progress percentage the epoch number and delete the previous line
        print("Progress: %d%%, Epoch: %d, Nets: %s" % (epoch/epochs*100, epoch, NETS), end='\r')     
        for i in range(iter):
            # sample batch random indices from the training data and set them as the batch
            idx = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_train_batch = X_train[idx].to(device)
            U_train_batch = U_train[idx].to(device)
            est.train(X_train_batch, U_train_batch)
        
        if epoch > 4:
            #sample batch random indices from the testing data and set them as the batch
            idx = np.random.choice(X_test.shape[0], batch_size, replace=False)
            X_test_batch = X_test[idx].to(device)
            U_test_batch = U_test[idx].to(device)
            est.eval(X_test_batch, U_test_batch, epoch)
            if epoch % 50 == 0 and epoch > 30:
                ename = "epoch %d.dat" % (epoch)
                fename = os.path.join(save_path, ename)
                est.save(fename)
