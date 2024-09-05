import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import os, copy
from tensorboardX import SummaryWriter
import datetime

date = datetime.datetime.now().strftime("%d.%h.%H.%M")
dirPath = os.path.dirname(os.path.realpath(__file__))
parentDirPath = os.path.dirname(dirPath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID_DIM = 1000
LR = 1e-6


class TORQUE_ESTIMATOR(nn.Module):
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
                
    def __init__(self, input_dim, output_dim):
        super(TORQUE_ESTIMATOR, self).__init__()

        self.encoder = nn.Sequential(
                    nn.Linear(input_dim, HID_DIM),
                    nn.LeakyReLU(),
                    nn.Linear(HID_DIM, HID_DIM),
                    nn.LeakyReLU(),
                    nn.Linear(HID_DIM, HID_DIM),
                    nn.LeakyReLU())
        
        self.encoder_sub = nn.Sequential(
                    nn.Linear(input_dim, HID_DIM),
                    nn.LeakyReLU(),
                    nn.Linear(HID_DIM, HID_DIM),
                    nn.LeakyReLU(),
                    nn.Linear(HID_DIM, output_dim),
                    nn.LeakyReLU())
        
        self.mu = nn.Linear(HID_DIM, output_dim)
        self.log_std = nn.Linear(HID_DIM, output_dim)
        
        self.encoder.apply(self.init_weights)
      
    def forward(self, state,training):
        # Non Gaussian
        # out = self.encoder_sub(state)
        # return out

        # Gaussian
        out = self.encoder(state)
        mu, log_std = self.mu(out), self.log_std(out)
        log_std = torch.clip(log_std, -2, 15)
        std = torch.exp(log_std)
        if training:
            torq = torch.normal(mu, std)
        else:
            torq = mu
        return torq


class ESTIMATOR(nn.Module):
    def __init__(self, input_dim, output_dim, save_run):
        super(ESTIMATOR, self).__init__()
        if save_run:
            self.writer = SummaryWriter(parentDirPath+'/data/runs/Summary/'+"_EST_"+"_"+"L_DIM_"+str(HID_DIM)+"_"\
            +"LR_"+str(LR)+"_"+date)

        self.est = TORQUE_ESTIMATOR(input_dim, output_dim).to(device)
        self.est_optimizer = optim.Adam(self.est.parameters(), lr=LR, weight_decay=1e-5)
        self.scheduler_e = optim.lr_scheduler.StepLR(self.est_optimizer, step_size=2000, gamma=0.1)
       
        self.train_step = 0
    
    def train(self, input_data, target):
        pred_torque = self.est(input_data, True)
        loss = F.mse_loss(pred_torque, target)
        self.est_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.est.parameters(), CG) 
        self.est_optimizer.step()
        # write the loss to tensorboard
        self.writer.add_scalar('MSE_loss/training',loss, self.train_step)
        
        self.train_step += 1

    def eval(self, input_data, target, epoch): # same as train, but without optimizer steps
        with torch.no_grad():
            pred_torque = self.est(input_data, False)
            loss = F.mse_loss(pred_torque, target)
            self.writer.add_scalar('MSE_loss/evaluation',loss, epoch)
            #calculate l1 loss  
            l1_loss = F.l1_loss(pred_torque, target)
            self.writer.add_scalar('L1_loss/evaluation',l1_loss, epoch)
    
    def cal_tau(self, input_data):
        with torch.no_grad():
            pred_torque = self.est(input_data, False)
        return pred_torque
     
    def save(self, filename):
        torch.save(self.est.state_dict(), filename + "_est")

    def load(self, filename):
        self.est.load_state_dict(torch.load(filename + "_est"))